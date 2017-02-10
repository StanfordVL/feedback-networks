#!/bin/bash
#
#SBATCH --job-name=our_v35a_cifar100
#SBATCH --output=our_v35a_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 20 -netType 'our_v35a' -save 'checkpoints_our_v35a' 2>&1 | tee log_our_v35a.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 50 -netType 'our_v35a' -save 'checkpoints_our_v35a_cond' -resume 'checkpoints_our_v35a' 2>&1 | tee log_our_v35a_cond.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 65 -netType 'our_v35a' -save 'checkpoints_our_v35a_cond2' -resume 'checkpoints_our_v35a_cond' 2>&1 | tee log_our_v35a_cond2.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 80 -netType 'our_v35a' -save 'checkpoints_our_v35a_cond3' -resume 'checkpoints_our_v35a_con2' 2>&1 | tee log_our_v35a_cond3.txt

#th main.lua -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 65 -netType 'our_v35a' -save 'checkpoints_our_v35a_cond2' -resume 'checkpoints_our_v35a_cond' 2>&1 | tee log_our_v35a_cond2.txt
#th main.lua -LR 0.01 -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 85 -netType 'our_v35a' -save 'checkpoints_our_v35a_cond3_LR0p01' -resume 'checkpoints_our_v35a_cond2' 2>&1 | tee log_our_v35a_cond3_LR0p01.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 65 -netType 'our_v35a' -save 'checkpoints_our_v35a_cond2_LR0p01' -resume 'checkpoints_our_v35a_cond' -LR 0.01 2>&1 | tee log_our_v35a_cond2_LR0p01.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 95 -netType 'our_v35a' -save 'checkpoints_our_v35a_cond2_LR0p001' -resume 'checkpoints_our_v35a_cond2_LR0p01' -LR 0.001 2>&1 | tee log_our_v35a_cond3_LR0p001.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 80 -netType 'our_v35a' -save 'checkpoints_our_v35a_cond2' -resume 'checkpoints_our_v35a_cond' 2>&1 | tee log_our_v35a_cond2.txt
