#!/bin/bash
#
#SBATCH --job-name=resnetX2_avgPool_cifar100
#SBATCH --output=resnetX2_avgPool_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128000
#SBATCH --gres gpu:16

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'resnetX2_avgPool' -save 'checkpoints_resnetX2_avgPool' 2>&1 | tee log_resnetX2_avgPool_20.txt
