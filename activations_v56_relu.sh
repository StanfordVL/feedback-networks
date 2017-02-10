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

NET_NAME='our_v56'
#CHECKPOINT='checkpoints_our_v56_relu_cond6_LR0p01'
CHECKPOINT='checkpoints_our_v56_cond6_LR0p01_best'

th main.lua -sequenceOut 'false' -seqLength 4 -testOnly 'true' -nGPU 1 -depth 20 -batchSize 128 -dataset cifar100 -netType $NET_NAME -resume $CHECKPOINT
