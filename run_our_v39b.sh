#!/bin/bash
#
#SBATCH --job-name=our_v39b_cifar100
#SBATCH --output=our_v39b_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 15 -netType 'our_v39b' -save 'checkpoints_our_v39bt8' 2>&1 | tee log_our_v39bt8.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 50 -netType 'our_v39b' -save 'checkpoints_our_v39bt8_cond' -resume 'checkpoints_our_v39bt8' 2>&1 | tee log_our_v39bt8_cond.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 40 -netType 'our_v39b' -save 'checkpoints_our_v39bt8_e40_cond' -resume 'checkpoints_our_v39bt8' 2>&1 | tee log_our_v39bt8_e40_cond.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 65 -netType 'our_v39b' -save 'checkpoints_our_v39bt8_cond2' -resume 'checkpoints_our_v39bt8_cond' 2>&1 | tee log_our_v39bt8_cond2.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 80 -netType 'our_v39b' -save 'checkpoints_our_v39bt8_cond3' -resume 'checkpoints_our_v39bt8_cond2' 2>&1 | tee log_our_v39bt8_cond3.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 90 -netType 'our_v39b' -save 'checkpoints_our_v39bt8_cond4_LR0p01' -resume 'checkpoints_our_v39bt8_cond3' -LR 0.01 2>&1 | tee log_our_v39bt8_cond4_LR0p01.txt
th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 120 -netType 'our_v39b' -save 'checkpoints_our_v39bt8_cond5_LR0p001' -resume 'checkpoints_our_v39bt8_cond4_LR0p01' -LR 0.001 2>&1 | tee log_our_v39bt8_cond5_LR0p001.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 65 -netType 'our_v39b' -save 'checkpoints_our_v39bt8_cond2_LR0p05' -resume 'checkpoints_our_v39bt8_cond' -LR 0.05 2>&1 | tee log_our_v39bt8_cond2_LR0p05.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 95 -netType 'our_v39b' -save 'checkpoints_our_v39bt8_cond3_LR0p01' -resume 'checkpoints_our_v39bt8_cond2_LR0p05' -LR 0.01 2>&1 | tee log_our_v39bt8_cond3_LR0p01.txt
