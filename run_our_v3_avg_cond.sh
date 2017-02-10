#!/bin/bash
#
#SBATCH --job-name=our_v3_avg_cifar100
#SBATCH --output=our_v3_avg_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'our_v3_avg' -save 'checkpoints_our_v3_avg_cond2_v2' -resume 'checkpoints_our_v3_avg_cond_v2' -LR 0.05 2>&1 | tee log_our_v3_avg_cond2_v2.txt
