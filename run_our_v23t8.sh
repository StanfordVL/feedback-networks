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

#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 15 -netType 'our_v23' -save 'checkpoints_our_v23t8' 2>&1 | tee log_our_v23t8.txt
th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 48 -netType 'our_v23' -save 'checkpoints_our_v23t8_e48_cond' -resume 'checkpoints_our_v23t8' 2>&1 | tee log_our_v23t8_e48_cond.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 70 -netType 'our_v23' -save 'checkpoints_our_v23t8_cond2' -resume 'checkpoints_our_v23t8_cond' 2>&1 | tee log_our_v23t8_cond2.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 100 -netType 'our_v23' -save 'checkpoints_our_v23t8_cond3' -resume 'checkpoints_our_v23t8_cond2' -LR 0.01 2>&1 | tee log_our_v23t8_cond3.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 100 -netType 'our_v23' -save 'checkpoints_our_v23t8_cond4' -resume 'checkpoints_our_v23t8_cond3' -LR 0.001 2>&1 | tee log_our_v23t8_cond4.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 80 -netType 'our_v23' -save 'checkpoints_our_v23t8_cond2_LR0p01' -resume 'checkpoints_our_v23t8_cond' -LR 0.01 2>&1 | tee log_our_v23t8_cond2_LR0p01.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 65 -netType 'our_v23' -save 'checkpoints_our_v23t8_cond2_LR0p05' -resume 'checkpoints_our_v23t8_cond' -LR 0.05 2>&1 | tee log_our_v23t8_cond2_LR0p05.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 95 -netType 'our_v23' -save 'checkpoints_our_v23t8_cond3_LR0p01' -resume 'checkpoints_our_v23t8_cond2_LR0p05' -LR 0.01 2>&1 | tee log_our_v23t8_cond3_LR0p01.txt
