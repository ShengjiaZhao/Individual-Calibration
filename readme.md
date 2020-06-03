# Introduction

This is the code that reproduces experiments in (paper)

# Requirements

The code is tested on python3.6 and pytorch 1.5. Also requires scipy, sklearn, PIL packages to run. 

# Running the code

To run the code without post-training recalibration, use

```
python train.py --gpu=0 
```

To apply post-training recalibration, use
```
python train.py --gpu=0 --recalibrate
```

You can also apply group recalibration for a certain feature, for example
```
python train.py --gpu=0 --recalibrate --group_idx=2 
```
recalibrates the subgroups partitioned by the second input feature. The partition currently is based on greater or less than the median. 

You can use the code in ```plot.ipynb``` to reproduce the calibration error comparison plot between different methods.