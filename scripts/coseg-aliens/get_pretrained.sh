#!/usr/bin/env bash

CHECKPOINT_DIR=$(dirname $0)/'../../checkpoints'

mkdir -p $CHECKPOINT_DIR && cd $CHECKPOINT_DIR
wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/checkpoints/coseg-aliens/coseg-aliens.pkl
echo "downloaded the checkpoint and putting it in: " $CHECKPOINT_DIR
