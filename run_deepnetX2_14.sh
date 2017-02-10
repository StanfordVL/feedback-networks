#!/bin/bash
#
#SBATCH --job-name=deepnetX2_13_cifar100
#SBATCH --output=deepnetX2_13_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'deepnetX2_14' -save 'checkpoints_deepnetX2_14' 2>&1 | tee log_deepnetX2_14_no_avgPool_no_dropout.txt
