#!/bin/bash
#
#SBATCH --job-name=our_v31_cifar100
#SBATCH --output=our_v31_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:2

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 30 -netType 'our_v31' -save 'checkpoints_our_v31' 2>&1 | tee log_our_v31.txt
