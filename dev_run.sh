#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# training param
DATASET='mnist' # mnist or cifar10
MODEL='lenet' # base, small, or large (for cifar10) || lenet, modelA, or modelB (for mnist)
LR=0.001
NUM_EPOCHS=3
BATCH_SIZE=32
DROP_P=0.5
PATIENCE=20
USE_MYDROPOUT=0

# neptune
NAME='exp-recap-1'
TAG='mnist-performance'

#for i in 1 2 3
#do
    python3 dev_main.py \
        --dataset=$DATASET \
        --model=$MODEL \
        --lr=$LR \
        --num_epochs=$NUM_EPOCHS \
        --batch_size=$BATCH_SIZE \
        --patience=$PATIENCE \
        --name=$NAME \
        --tag=$TAG \
        --drop_p=$DROP_P \
        --use_mydropout=$USE_MYDROPOUT \
#done
