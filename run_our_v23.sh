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

#th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 65 -netType 'our_v23' -save 'checkpoints_our_v23_cond5_LR0p05' -resume 'checkpoints_our_v23_cond4' -LR 0.05 2>&1 | tee log_our_v23_cond5_LR0p05.txt
#th main.lua -LR 0.05 -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 75 -netType 'our_v23' -save 'checkpoints_our_v23_cond6_LRp0p5' -resume 'checkpoints_our_v23_cond5' 2>&1 | tee log_our_v23_cond6_LR0p05.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 50 -netType 'our_v23' -save 'checkpoints_our_v23_cond4' -resume 'checkpoints_our_v23_cond3' 2>&1 | tee log_our_v23_cond4.txt
th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 90 -netType 'our_v23' -save 'checkpoints_our_v23_cond6' -resume 'checkpoints_our_v23_cond5' -LR 0.01 2>&1 | tee log_our_v23_cond6.txt
