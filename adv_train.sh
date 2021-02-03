#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

# training param
EPSILON=0.3
ITERATION=40

LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.0005
MAX_EPOCHS=50
BATCH_SIZE=64
USE_SCHEDULER=1
STEP_SIZE=10
GAMMA=0.5
SAVEPATH='mydict.pth'

# neptune
NAME='21.02.03.exp1'
TAG='normal'

for i in 1 2 3
do
    python3 adv_train.py \
        --lr=$LR \
        --max_epochs=$MAX_EPOCHS \
        --batch_size=$BATCH_SIZE \
        --use_scheduler=$USE_SCHEDULER \
        --step_size=$STEP_SIZE \
        --gamma=$GAMMA \
        --epsilon=$EPSILON \
        --momentum=$MOMENTUM \
        --savepath=$SAVEPATH \
        --name=$NAME \
        --tag=$TAG \
        --weight_decay=$WEIGHT_DECAY \
        --iteration=$ITERATION
done
