#!/bin/bash
#
#SBATCH --job-name=resnet_32
#SBATCH --output=resnet_32_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:2

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -depth 32 -batchSize 128 -dataset cifar100 -nEpochs 120 -save 'checkpoints_resnet32' -resume 'checkpoints_resnet32_2' 2>&1 | tee log_resnet_32_2.txt
