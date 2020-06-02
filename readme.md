# Introduction

This is the code that reproduces experiments in (paper)


# Requirements

The code is tested on python3.6 and pytorch 2.1. Also requires scipy, sklearn, PIL packages to run. 

# Running the code

To run the code without post-training recalibration, use

```
python train.py --gpu=0 
```

To apply post-training recalibration, use
```
python train.py --gpu=0 --recalibrate
```
