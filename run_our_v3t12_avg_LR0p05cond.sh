#!/bin/bash
#
#SBATCH --job-name=our_v3t12_avg_cifar100
#SBATCH --output=our_v3t12_avg_LR0p05cond_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:2

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 80 -netType 'our_v3_avg' -save 'checkpoints_our_v3t12_avg_LR0p05cond' -resume 'checkpoints_our_v3t12_avg' -LR 0.05 2>&1 | tee log_our_v3t12_avg_v2_LR0p05cond.txt
