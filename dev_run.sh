#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

# training param
DATASET='mnist' # mnist or cifar10
if [ $DATASET = 'mnist' ]
then
    EPSILON=0.3
elif [ $DATASET = 'cifar10' ]
then
    EPSILON=0.03137
else
    echo 'dataset must be either mnist of cifar10'
fi

MODEL='lenet' # base, small, or large (for cifar10) || lenet, modelA, or modelB (for mnist)
LR=$2
WEIGHT_DECAY=0.0005
NUM_EPOCHS=50
BATCH_SIZE=64
DROP_P=0.3
PATIENCE=2000
USE_STEP_POLICY=$3
STEP_SIZE=60
GAMMA=0.2
USE_MYDROPOUT=0
ADV_TEST_OUT_PATH='./result/adv_dummy.pkl'
ITERATION=40

# neptune
NAME='21.01.26.exp2.debug'
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
        --use_step_policy=$USE_STEP_POLICY \
        --step_size=$STEP_SIZE \
        --gamma=$GAMMA \
        --weight_decay=$WEIGHT_DECAY \
        --adv_test_out_path=$ADV_TEST_OUT_PATH \
        --iteration=$ITERATION \
        --epsilon=$EPSILON
done
