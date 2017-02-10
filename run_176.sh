#!/bin/bash
#
#SBATCH --job-name=resnet_176
#SBATCH --output=resnet_176_no_dropout_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:2

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -depth 176 -batchSize 128 -dataset cifar100 -nEpochs 120 -save 'checkpoints_resnet176' -nGPU 2 2>&1 | tee log_resnet_176_no_dropout.txt
