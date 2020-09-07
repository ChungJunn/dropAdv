export CUDA_VISIBLE_DEVICES=$1
LR=0.001
NUM_EPOCHS=2000
LOG_INTERVAL=1000
BATCH_SIZE=64
EPSILON=0.15
ALPHA=0.5
DROP_P=$2
PATIENCE=20
ADV_PATIENCE=40
ADV_TRAIN=$3

NAME=$4
TAG='tag'

IS_DNN=0

python3 cifar10.py \
    --lr=$LR \
    --num_epochs=$NUM_EPOCHS \
    --log_interval=$LOG_INTERVAL \
    --batch_size=$BATCH_SIZE \
    --epsilon=$EPSILON \
    --alpha=$ALPHA \
    --patience=$PATIENCE \
    --name=$NAME \
    --tag=$TAG \
    --drop_p=$DROP_P \
    --is_dnn=$IS_DNN \
    --adv_patience=$ADV_PATIENCE \
    --adv_train=$ADV_TRAIN
