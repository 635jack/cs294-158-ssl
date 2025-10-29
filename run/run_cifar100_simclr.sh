#!/bin/sh

# 1-2 GPUs - CIFAR100 self-supervised learning avec SimCLR
python train_self_supervised_task.py -d cifar100 -t simclr -b 128 -e 200

