# Getting Started


1. This repository consists of the experiments to reproduce results in the [paper](https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization) in PyTorch

2. Implementation of yogi optimizer proposed in the paper.

# How to run

## Step-1:

```
git clone https://github.com/avinashsai/Adaptive-Nonconvex-Optimization.git

cd PyTorch
```
## Step-2:

***Note: This implementation currently supports Image Classification only***

```
python main.py --task (image classification (default) | auto encoders | machine translation) 
               --nocuda (Disable or Enable cuda (default: False)
               --runs (Number of runs to report results (default: 6)
```
## Step-3:

If task is ***image classification***, user will be prompted for an input whether to use ResNet20 or ResNet50

```
If ResNet20 Enter 1

If ResNet50 Enter 2
```

# How to Use Optimizer

Copy yogi.py file and run following lines:

```
from yogi import *

yogi = Yogi(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-3, 
                initial_accumulator=1e-6)
```
