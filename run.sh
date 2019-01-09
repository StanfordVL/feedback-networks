#!/bin/bash

th main.lua -seqLength 4 -sequenceOut 'true' -nGPU 1 -depth 20 -batchSize 64 -dataset cifar100 -nEpochs 120 -netType 'feedback_48' -save 'checkpoints_feedback_48' 2>&1 | tee log_feedback_48.txt

