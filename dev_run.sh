#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

# training param
DATASET='cifar10' # mnist or cifar10
MODEL='wide-resnet' # base, small, or large (for cifar10) || lenet, modelA, or modelB (for mnist)
LR=$2
NUM_EPOCHS=1000
BATCH_SIZE=64
DROP_P=0.25
PATIENCE=20
USE_STEP_POLICY=$3
USE_MYDROPOUT=0

# neptune
NAME='21.01.25.exp3'
TAG='none'

for i in 1 2 3
do
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
        --use_step_policy=$USE_STEP_POLICY
done
