#!/bin/bash
#
#SBATCH --job-name=our_v34Pool_cifar100
#SBATCH --output=our_v34Pool_cifar100-%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1

ml torch/20160805-4bfc2da protobuf/2.6.1
ml CUDA/7.5.18 cuDNN/5.0-CUDA-7.5.18

th main.lua -seqLength 8 -sequenceOut 'true' -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 20 -netType 'our_v34Pool' -save 'checkpoints_our_v34Pool' 2>&1 | tee log_our_v34Pool.txt
#th main.lua -seqLength 8 -sequenceOut 'true' -nGPU 1 -depth 20 -batchSize 96 -dataset cifar100 -nEpochs 50 -netType 'our_v34Pool' -save 'checkpoints_our_v34_condPool' -resume 'checkpoints_our_v34Pool' 2>&1 | tee log_our_v34Pool_cond.txt
