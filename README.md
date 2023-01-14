# Hand State RCNN

## Install

```
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Demo

We have provided the trained weights on outputs/ folder (with the training log). It achieves 84.9 AP@50 on 100 DOH test set.

See hsrcnn.py, modify the video path and our program will produce a new video of hand interaction detection.

## Data

1. download zips from 100DOH: https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/, unzip them to a created '100doh' folder.  

2. python dohstate.py # prepare data

## Train

```
CUDA_VISIBLE_DEVICES=6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=2 train.py --batch-size 16 --lr 0.01 --sync-bn --amp --output-dir outputs/ms_bs2x16_syncbn_amp > outputs/bs2x16_syncbn_amp/log.txt
```

## Evaluation

```
torchrun --nproc_per_node=8 train.py --batch-size 32 --test-only --sync-bn --amp --resume outputs/ms_bs2x16_lr1e-2_12e_syncbn_amp/model_11.pth
```

