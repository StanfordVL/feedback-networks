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

th main.lua -seqLength 2 -testOnly 'true' -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -netType 'our_v39b' -resume 'checkpoints_our_v39bt8_cond3_LR0p01'
