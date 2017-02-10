#!/bin/bash
#
#SBATCH --job-name=our_v37_cifar100
#SBATCH --output=our_v37_cifar100-%j.txt
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:2

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 20 -netType 'our_v37' -save 'checkpoints_our_v37' 2>&1 | tee log_our_v37.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 50 -netType 'our_v37' -save 'checkpoints_our_v37_cond' -resume 'checkpoints_our_v37' 2>&1 | tee log_our_v37_cond.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 65 -netType 'our_v37' -save 'checkpoints_our_v37_cond2_LR0p01' -resume 'checkpoints_our_v37_cond' -LR 0.01 2>&1 | tee log_our_v37_cond2_LR0p01.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 60 -netType 'our_v37' -save 'checkpoints_our_v37_cond2' -resume 'checkpoints_our_v37_cond' 2>&1 | tee log_our_v37_cond2.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 80 -netType 'our_v37' -save 'checkpoints_our_v37_cond3' -resume 'checkpoints_our_v37_cond2' 2>&1 | tee log_our_v37_cond3.txt
th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'our_v37' -save 'checkpoints_our_v37_cond4_LR0p01' -resume 'checkpoints_our_v37_cond3' 2>&1 | tee log_our_v37_cond4_LR0p01.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 80 -netType 'our_v37' -save 'checkpoints_our_v37_cond3_LR0p01' -resume 'checkpoints_our_v37_cond2' -LR 0.01 2>&1 | tee log_our_v37_cond3_LR0p01.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 80 -netType 'our_v37' -save 'checkpoints_our_v37_cond3_LR0p01' -resume 'checkpoints_our_v37_cond2_LR0p01' -LR 0.01 2>&1 | tee log_our_v37_cond3_LR0p01.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 80 -netType 'our_v37' -save 'checkpoints_our_v37_cond2' -resume 'checkpoints_our_v37_cond' 2>&1 | tee log_our_v37_cond2.txt
