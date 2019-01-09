


# Feedback Networks http://feedbacknet.stanford.edu/ 

Paper: Feedback Networks, CVPR 2017.

Amir R. Zamir*,Te-Lin Wu*, Lin Sun, William B. Shen, Bertram E. Shi, Jitendra Malik, Silvio Savarese. 

## Feedback Networks training in Torch
============================

## Requirements
Code adopted and modified from [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
See the [installation instructions](INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4 or v5](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
- Install [rnn](https://github.com/Element-Research/rnn) the Element-Research RNN library for Torch
- Download the [CIFAR10/100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (binary file) and put it under folder gen/

If you already have Torch installed, update `nn`, `cunn`, and `cudnn`.

## Training

The training scripts come with several options, which can be listed with the `--help` flag.
```bash
th main.lua --help
```

To run the training, see the example run.sh, explanations below:
```bash
th main.lua -seqLength [number of feedback iterations] -sequenceOut [true for feedback false for recurrence inference] -nGPU [number of GPU]
-depth [20 to bypass] -batchSize [batch size] -dataset [cifar100] -nEpochs [number of epochs to train]
-netType [the model under models/ directory] -save [checkpoints directory to save the model] -resume [checkpoints directory to restore the model]
```

## Testing

To run the testing, simply assign a directory of where the checkpoints are saved and turn of the testOnly flag and specify the model path as follows:
```bash
-testOnly 'true' -resume [checkpoints directory to restore the model]
```

## Using your own criterion

You can write your own criterion and store it under the directory lib/, and require them in the models/init.lua
Add another options in the opts.lua to use them while running a script, for example
```lua
cmd:option('-coarsefine', 'false', 'If using this criterion or not')
opt.coarsefine = opt.coarsefine ~= 'false'
```
In the bash script add
```bash
-coarsefine 'true'
```

## Writing your own model

You can develop your own model and store in under models/, as an exmaple model of ours, models/feedback_48.lua
Modify the code below the following lines within the code block, and set the netType in your running bash script or command
to the name of the model you develop:
```lua
elseif opt.dataset == 'cifar100' then
   -- Model type specifies number of layers for CIFAR-100 model
```

