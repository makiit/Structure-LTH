#!/usr/bin/env bash
#SBATCH --job-name=places-train
#### Change account below
#SBATCH --account=ddp390
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=010:00:00
#SBATCH --output=output.o%j.%N

# load the environments needed

module purge
module load slurm
module load gpu
module load cuda10.2/toolkit/10.2.89
module list

nvidia-smi
nvcc -V

# python train.py --config=configs/convnet4/mini-imagenet/5_way_5_shot/train_reproduce.yaml
# python test.py --config=configs/convnet4/mini-imagenet/5_way_5_shot/test_reproduce.yaml
 # python -u main_imp.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.2_s1_rewind_16 --init pretrained_model/res18_cifar10_1_init.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --epoch 160 --decreasing_lr 80,120 --rewind_epoch 16 --weight_decay 1e-4 --batch_size 128
i=1
python -u main_eval_regroup_retrain.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir  --pretrained resnet18_cifar10_lt_0.2_s1_rewind_16/1checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.2_s1_rewind_16/${i}checkpoint.pth.tar --fc --prune-type lt --seed 1 --epoch 160 --decreasing_lr 80,120 --weight_decay 1e-4 --batch_size 128 --lr 0.1 
# run your code
