#!/bin/bash
#
#SBATCH --job-name=our_v3_avg_cifar100
#SBATCH --output=our_v3_avg_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:2

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -nGPU 1 -depth 20 -batchSize 64 -dataset cifar100 -nEpochs 120 -netType 'our_v3_avg' -save 'checkpoints_our_v3t12_avg_cond4' -resume 'checkpoints_our_v3t12_avg_cond3' -LR 0.0005 2>&1 | tee log_our_v3t12_avg_v2_cond4.txt
