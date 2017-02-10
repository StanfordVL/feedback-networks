#!/bin/bash
#
#SBATCH --job-name=our_v35t8_cifar100
#SBATCH --output=our_v35t8_cifar100-%j.txt
#SBATCH --time=0-03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:2

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 20 -netType 'our_v35t8' -save 'checkpoints_our_v35t8' 2>&1 | tee log_our_v35t8.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 50 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond' -resume 'checkpoints_our_v35t8' 2>&1 | tee log_our_v35t8_cond.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 65 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond2' -resume 'checkpoints_our_v35t8_cond' 2>&1 | tee log_our_v35t8_cond2.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 65 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond2' -resume 'checkpoints_our_v35t8_cond' 2>&1 | tee log_our_v35t8_cond2.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 80 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond3' -resume 'checkpoints_our_v35t8_cond2' 2>&1 | tee log_our_v35t8_cond3.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 90 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond4' -resume 'checkpoints_our_v35t8_cond3' 2>&1 | tee log_our_v35t8_cond4.txt
th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 100 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond5' -resume 'checkpoints_our_v35t8_cond4' 2>&1 | tee log_our_v35t8_cond5.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 110 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond6' -resume 'checkpoints_our_v35t8_cond5' 2>&1 | tee log_our_v35t8_cond6.txt
#th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 115 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_condcond7_LR0p01' -resume 'checkpoints_our_v35t8_cond6' 2>&1 | tee log_our_v35t8_condcond7_LR0p01.txt
#th main.lua -LR 0.001 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 115 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_condcond8_LR0p001' -resume 'checkpoints_our_v35t8_condcond7_LR0p01' 2>&1 | tee log_our_v35t8_condcond8_LR0p001.txt

#th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 100 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond6_new_LR0p01' -resume 'checkpoints_our_v35t8_cond5' 2>&1 | tee log_our_v35t8_cond6_new_LR0p01.txt

#th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 110 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond5_LR0p01' -resume 'checkpoints_our_v35t8_cond4' 2>&1 | tee log_our_v35t8_cond5_LR0p01.txt
#th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 110 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond6_LR0p01' -resume 'checkpoints_our_v35t8_cond5_LR0p01' 2>&1 | tee log_our_v35t8_cond6_LR0p01.txt
#th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 110 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond7_LR0p01' -resume 'checkpoints_our_v35t8_cond6_LR0p01' 2>&1 | tee log_our_v35t8_cond7_LR0p01.txt
#th main.lua -LR 0.001 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond8_LR0p001' -resume 'checkpoints_our_v35t8_cond7_LR0p01' 2>&1 | tee log_our_v35t8_cond8_LR0p001.txt

#th main.lua -weightDecay 1e-2 -momentum 0.96 -LR 0.001 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond8_LR0p01' -resume 'checkpoints_our_v35t8_cond7_LR0p01' 2>&1 | tee log_our_v35t8_cond8_LR0p01.txt
#th main.lua -weightDecay 1e-2 -momentum 0.1 -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond8_LR0p01' -resume 'checkpoints_our_v35t8_cond7_LR0p01' 2>&1 | tee log_our_v35t8_cond8_LR0p01_2.txt

#th main.lua -weightDecay 1e-6 -momentum 0 -LR 0.0001 -nGPU 1 -depth 20 -batchSize 64 -dataset cifar100 -nEpochs 120 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond9_LR0p001' -resume 'checkpoints_our_v35t8_cond8_LR0p001' 2>&1 | tee log_our_v35t8_cond9_LR0p001_new.txt
#th main.lua -weightDecay 1e-1 -momentum 0.001 -LR 0.0001 -nGPU 1 -depth 20 -batchSize 64 -dataset cifar100 -nEpochs 120 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond10_LR0p0001' -resume 'checkpoints_our_v35t8_cond9_LR0p001' 2>&1 | tee log_our_v35t8_cond10_LR0p0001_new.txt
#th main.lua -weightDecay 1e-2 -momentum 0 -LR 0.0001 -nGPU 1 -depth 20 -batchSize 64 -dataset cifar100 -nEpochs 120 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond10_LR0p0001' -resume 'checkpoints_our_v35t8_cond7_LR0p01' 2>&1 | tee log_our_v35t8_cond10_LR0p0001_new.txt

#th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 100 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond4_LR0p01' -resume 'checkpoints_our_v35t8_cond3' 2>&1 | tee log_our_v35t8_cond4_LR0p01.txt
#th main.lua -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 100 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond4' -resume 'checkpoints_our_v35t8_cond3' 2>&1 | tee log_our_v35t8_cond4.txt
#th main.lua -LR 0.0001 -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 140 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond6_LR0p0001' -resume 'checkpoints_our_v35t8_cond5_LR0p001' 2>&1 | tee log_our_v35t8_cond6_LR0p0001.txt
#th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond5_LR0p01' -resume 'checkpoints_our_v35t8_cond4' 2>&1 | tee log_our_v35t8_cond5_LR0p01.txt
#th main.lua -LR 0.01 -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 120 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond42_LR0p01' -resume 'checkpoints_our_v35t8_cond4_LR0p01' 2>&1 | tee log_our_v35t8_cond42_LR0p01.txt
#th main.lua -LR 0.001 -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 150 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond43_LR0p001' -resume 'checkpoints_our_v35t8_cond42_LR0p01' 2>&1 | tee log_our_v35t8_cond43_LR0p001.txt
#th main.lua -LR 0.001 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 140 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond6_LR0p001' -resume 'checkpoints_our_v35t8_cond5_LR0p01' 2>&1 | tee log_our_v35t8_cond6_LR0p001.txt
#th main.lua -LR 0.01 -nGPU 2 -depth 20 -batchSize 128 -dataset cifar100 -nEpochs 90 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond3_LR0p01' -resume 'checkpoints_our_v35t8_cond2' 2>&1 | tee log_our_v35t8_cond3_LR0p01.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 65 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond2_LR0p01' -resume 'checkpoints_our_v35t8_cond' -LR 0.01 2>&1 | tee log_our_v35t8_cond2_LR0p01.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 95 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond2_LR0p001' -resume 'checkpoints_our_v35t8_cond2_LR0p01' -LR 0.001 2>&1 | tee log_our_v35t8_cond3_LR0p001.txt
#th main.lua -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 80 -netType 'our_v35t8' -save 'checkpoints_our_v35t8_cond2' -resume 'checkpoints_our_v35t8_cond' 2>&1 | tee log_our_v35t8_cond2.txt
