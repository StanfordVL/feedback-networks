#!/bin/bash
#
#SBATCH --job-name=our_v23_cifar100
#SBATCH --output=our_v23_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:2

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

#th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 50 -netType 'our_v29' -save 'checkpoints_our_v29_cond' -resume 'checkpoints_our_v29' 2>&1 | tee log_our_v29_cond.txt
th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 100 -netType 'our_v29' -save 'checkpoints_our_v29_cond2' -resume 'checkpoints_our_v29_cond' -LR 0.01 2>&1 | tee log_our_v29_cond2.txt
