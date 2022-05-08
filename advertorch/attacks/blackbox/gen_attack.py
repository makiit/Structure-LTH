# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Optional
from math import inf

import torch
import torch.nn.functional as F

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

from .utils import _check_param, _flatten, _make_projector


def gen_attack_score(output, target, targeted=False, buffer=1e-5):
    """
    Fitness used for GenAttack
    """
    n_class = output.shape[-1]
    y_onehot = F.one_hot(target, num_classes=n_class)

    pos_score = torch.log((y_onehot[:, None, :] * output + buffer).sum(-1))
    neg_score = torch.log((1 - y_onehot)[:, None, :] * output + buffer).sum(-1)
    score = pos_score - neg_score

    if not targeted:
        score = -score

    return score


def compute_fitness(predict_fn, loss_fn, adv_pop, y, targeted=False):
    """
    Compute fitness for the population
    """
    # population shape: [B, N, F]
    n_batch, n_samples, n_dim = adv_pop.shape
    # reshape to [B * N, F]
    adv_pop = adv_pop.reshape(-1, n_dim)
    # output shape: [B * N, C]
    probs = predict_fn(adv_pop)

    # reshape to [B, N, C]
    probs = probs.reshape(n_batch, n_samples, -1)
    # outputs shape: [B, N]
    fitness = loss_fn(probs, y, targeted=targeted)

    return fitness


def crossover(p1, p2, probs):
    """
    Mate parents (p1, p2) to produce members of the next generation.
    Children are generated by selecting features from either parent.
    Select from p1 with the probabilties in probs.
    """
    u = torch.rand(*p1.shape)
    return torch.where(probs[:, :, None] > u, p1, p2)


def selection(pop_t, fitness, tau):
    """
    Select individuals in the population according to their fitness.
    These individuals become parents, which produce children via crossover.
    """
    n_batch, nb_samples, n_dim = pop_t.shape

    probs = F.softmax(fitness / tau, dim=1)
    cum_probs = probs.cumsum(-1)
    # Edge case, u1 or u2 is greater than max(cum_probs)
    cum_probs[:, -1] = 1. + 1e-7

    # parents: instead of selecting one elite, select two, and generate
    # a new child to create a population around
    # do this multiple times, for each N

    # sample parent 1 from pop_t according to probs (multinomial)
    # sample parent 2 from pop_t according to probs (multinomial)
    u1, u2 = torch.rand(2, n_batch, nb_samples)

    # out of the original N samples, we draw another N samples
    # this requires us to compute the following broadcasted comparison
    p1ind = -((cum_probs[:, :, None] > u1[:, None, :]
               ).long()).sum(1) + nb_samples
    p2ind = -((cum_probs[:, :, None] > u2[:, None, :]
               ).long()).sum(1) + nb_samples

    parent1 = torch.gather(
        pop_t, dim=1, index=p1ind[:, :, None].expand(-1, -1, n_dim)
    )

    parent2 = torch.gather(
        pop_t, dim=1, index=p2ind[:, :, None].expand(-1, -1, n_dim)
    )

    fp1 = torch.gather(fitness, dim=1, index=p1ind)
    fp2 = torch.gather(fitness, dim=1, index=p2ind)
    crossover_prob = fp1 / (fp1 + fp2)

    return crossover(parent1, parent2, crossover_prob)


def mutation(pop_t, alpha, rho, eps):
    """
    Add random noise to the population to explore the search space.

    Alpha controls the scale of the noise, rho controls the number of features
    that are perturbed.
    """
    # alpha and eps both have shape [B]
    perturb_noise = (2 * torch.rand(*pop_t.shape) - 1)
    perturb_noise = perturb_noise * alpha[:, None, None] * eps[:, None, None]

    mask = (torch.rand(*pop_t.shape) > rho[:, None, None]).float()

    return pop_t + mask * perturb_noise


class GenAttackScheduler():
    """
    Parameter scaling for GenAttack.  Decrease mutation rate and range when
    search is detected to be stuck.

    For more details, see section 4.1.2 of https://arxiv.org/abs/1805.11090.
    """

    def __init__(
        self, x, alpha_init=0.4, rho_init=0.5, decay=0.9,
        rho_min=0.1, alpha_min=0.15
    ):
        n_batch = x.shape[0]

        self.n_batch = n_batch
        self.crit = 1e-5

        self.best_val = torch.zeros(n_batch).to(x.device)
        self.num_i = torch.zeros(n_batch).to(x.device)
        self.num_plateaus = torch.zeros(n_batch).to(x.device)

        self.rho_min = rho_min * torch.ones(n_batch).to(x.device)
        self.alpha_min = alpha_min * torch.ones(n_batch).to(x.device)

        self.zeros = torch.zeros_like(self.num_i)

        self.alpha_init = alpha_init
        self.rho_init = rho_init
        self.decay = decay

        self.alpha = alpha_init * torch.ones(n_batch).to(x.device)
        self.rho = rho_init * torch.ones(n_batch).to(x.device)

    def update(self, elite_val):
        stalled = abs(elite_val - self.best_val) <= self.crit
        self.num_i = torch.where(stalled, self.num_i + 1, self.zeros)
        new_plateau = (self.num_i % 100 == 0) & (self.num_i != 0)
        self.num_plateaus = torch.where(
            new_plateau, self.num_plateaus + 1, self.num_plateaus
        )

        # update alpha and rho
        self.rho = torch.maximum(
            self.rho_min, self.rho_init * self.decay ** self.num_plateaus
        )
        self.alpha = torch.maximum(
            self.alpha_min, self.alpha_init * self.decay ** self.num_plateaus
        )

        self.best_val = torch.maximum(elite_val, self.best_val)


def gen_attack(
    predict_fn, loss_fn, x, y, eps, projector, nb_samples=100, nb_iter=40,
    tau=0.1, alpha_init=0.4, rho_init=0.5, decay=0.9,
    pop_init=None, scheduler=None, targeted=False
):
    """
    Use a genetic algorithm to iteratively maximize the loss over the input,
    while staying within eps of the original input (using a projector).

    Used as part of GenAttack.

    :param predict: forward pass function.
    :param loss_fn: loss function
        - must accept tensors of shape [nbatch, pop_size, ndim]
    :param x: input tensor.
    :param y: label tensor.
        - if None and self.targeted=False, compute y as predicted
        labels.
        - if self.targeted=True, then y must be the targeted labels.
    :param eps: maximum distortion.
    :param projector: function to project the perturbation into the eps-ball
        - must accept tensors of shape [nbatch, pop_size, ndim]
    :param nb_samples: population size (default 100)
    :param nb_iter: number of iterations (default 40)
    :param tau: sampling temperature (default 0.1)
    :param alpha_init: initial mutation range (default 0.4)
    :param rho_init: initial probability for mutation (default 0.5)
    :param decay: decay param for scheduler (default 0.9)
    :param pop_init: initial population for genetic alg (default None)
    :param scheduler: initial state of scheduler(default None)
    :param targeted: if the attack is targeted (default False)
    """
    n_batch, n_dim = x.shape

    # [B,F]
    if pop_init is None:
        # Sample from Uniform(-1, 1)
        # shape: [B, N, F]
        pop_t = 2 * torch.rand(n_batch, nb_samples, n_dim) - 1
        # Sample from Uniform(-eps, eps)
        pop_t = eps[:, None, None] * pop_t
        pop_t = pop_t.to(x.device)
    else:
        pop_t = pop_init.clone()

    if scheduler is None:
        scheduler = GenAttackScheduler(x, alpha_init, rho_init, decay)

    inds = torch.arange(n_batch).to(x.device)

    for _ in range(nb_iter):
        adv = x[:, None, :] + pop_t
        # shape: [B, N]
        fitness = compute_fitness(
            predict_fn, loss_fn, adv, y, targeted=targeted)
        # shape: [1, B, 1]
        elite_val, elite_ind = fitness.max(-1)
        # shape: [B, F]
        elite_adv = adv[inds, elite_ind, :]

        # select which members will move onto the next generation
        # shape: [B, N]
        children = selection(pop_t, fitness, tau)

        # apply mutations and clipping
        # add mutated child to next generation (ie update pop_t)
        pop_t = mutation(children, scheduler.alpha, scheduler.rho, eps)
        pop_t = projector(pop_t)

        # Update params based on plateaus
        scheduler.update(elite_val)

    return elite_adv, pop_t, scheduler


class GenAttack(Attack, LabelMixin):
    """
    Runs GenAttack https://arxiv.org/abs/1805.11090

    Disclaimers: Note that GenAttack assumes the model outputs
    normalized probabilities.  Moreover, computations are broadcasted,
    so it is advisable to use smaller batch sizes when nb_samples is
    large.

    Hyperparams: alpha (mutation range), rho (mutation probability),
    and tau (temperature) all control exploration.

    Alpha and rho are adapted using GenAttackScheduler.

    :param predict: forward pass function.
    :param eps: maximum distortion.
    :param order: the order of maximum distortion (inf or 2)
    :param loss_fn: loss function (default None, GenAttack uses its own loss)
    :param nb_samples: population size (default 100)
    :param nb_iter: number of iterations (default 40)
    :param tau: sampling temperature (default 0.1)
    :param alpha_init: initial mutation range (default 0.4)
    :param rho_init: initial probability for mutation (default 0.5)
    :param decay: decay param for scheduler (default 0.9)
    :param clip_min: mininum value per input dimension (default 0.)
    :param clip_max: mininum value per input dimension (default 1.)
    :param targeted: if the attack is targeted (default False)
    """

    def __init__(
        self, predict, eps: float, order,
        loss_fn=None,
        nb_samples=100,
        nb_iter=40,
        tau=0.1,
        alpha_init=0.4,
        rho_init=0.5,
        decay=0.9,
        clip_min=0., clip_max=1.,
        targeted: bool = False
    ):
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        super().__init__(predict, gen_attack_score, clip_min, clip_max)

        self.eps = eps
        self.order = order
        self.nb_samples = nb_samples
        self.nb_iter = nb_iter
        self.targeted = targeted

        self.alpha_init = alpha_init
        self.rho_init = rho_init
        self.decay = decay
        self.tau = tau

    def perturb(  # type: ignore
        self,
        x: torch.FloatTensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.FloatTensor:
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)
        shape, flat_x = _flatten(x)
        data_shape = tuple(shape[1:])

        # [B]
        eps = _check_param(self.eps, x.new_full((x.shape[0],), 1), 'eps')
        # [B, F]
        clip_min = _check_param(self.clip_min, flat_x, 'clip_min')
        clip_max = _check_param(self.clip_max, flat_x, 'clip_max')

        def f(x):
            new_shape = (x.shape[0],) + data_shape
            input = x.reshape(new_shape)
            return self.predict(input)

        projector = _make_projector(
            eps, self.order, flat_x, clip_min, clip_max
        )

        elite_adv, _, _ = gen_attack(
            predict_fn=f, loss_fn=self.loss_fn, x=flat_x, y=y,
            eps=eps, projector=projector,
            nb_samples=self.nb_samples, nb_iter=self.nb_iter, tau=self.tau,
            alpha_init=self.alpha_init, rho_init=self.rho_init,
            decay=self.decay, pop_init=None, scheduler=None,
            targeted=self.targeted
        )

        elite_adv = elite_adv.reshape(shape)

        return elite_adv


class LinfGenAttack(GenAttack):
    """
    GenAttack with order=inf

    :param predict: forward pass function.
    :param eps: maximum distortion.
    :param loss_fn: loss function (default None, GenAttack uses its own loss)
    :param nb_samples: population size (default 100)
    :param nb_iter: number of iterations (default 40)
    :param tau: sampling temperature (default 0.1)
    :param alpha_init: initial mutation range (default 0.4)
    :param rho_init: initial probability for mutation (default 0.5)
    :param decay: decay param for scheduler (default 0.9)
    :param clip_min: mininum value per input dimension (default 0.)
    :param clip_max: mininum value per input dimension (default 1.)
    :param targeted: if the attack is targeted (default False)
    """

    def __init__(
        self, predict, eps: float,
        loss_fn=None,
        nb_samples=100,
        nb_iter=40,
        tau=0.1,
        alpha_init=0.4,
        rho_init=0.5,
        decay=0.9,
        clip_min=0., clip_max=1.,
        targeted: bool = False
    ):
        super(LinfGenAttack, self).__init__(
            predict=predict, eps=eps, loss_fn=loss_fn, nb_samples=nb_samples,
            nb_iter=nb_iter, tau=tau, order=inf, alpha_init=alpha_init,
            rho_init=rho_init, decay=decay, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted
        )


class L2GenAttack(GenAttack):
    """
    GenAttack with order=2

    :param predict: forward pass function.
    :param eps: maximum distortion.
    :param loss_fn: loss function (default None, GenAttack uses its own loss)
    :param nb_samples: population size (default 100)
    :param nb_iter: number of iterations (default 40)
    :param tau: sampling temperature (default 0.1)
    :param alpha_init: initial mutation range (default 0.4)
    :param rho_init: initial probability for mutation (default 0.5)
    :param decay: decay param for scheduler (default 0.9)
    :param clip_min: mininum value per input dimension (default 0.)
    :param clip_max: mininum value per input dimension (default 1.)
    :param targeted: if the attack is targeted (default False)
    """

    def __init__(
        self, predict, eps: float,
        loss_fn=None,
        nb_samples=100,
        nb_iter=40,
        tau=0.1,
        alpha_init=0.4,
        rho_init=0.5,
        decay=0.9,
        clip_min=0., clip_max=1.,
        targeted: bool = False
    ):
        super(L2GenAttack, self).__init__(
            predict=predict, eps=eps, loss_fn=loss_fn, nb_samples=nb_samples,
            nb_iter=nb_iter, tau=tau, order=2, alpha_init=alpha_init,
            rho_init=rho_init, decay=decay, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted
        )
