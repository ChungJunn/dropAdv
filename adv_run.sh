SEED=93
LR=0.001
NUM_EPOCHS=1000
LOG_INTERVAL=1000
BATCH_SIZE=64
EPSILON=0.05
ALPHA=0.5
DROP_P=$1
PATIENCE=20
USE_ADV_TRAIN=$2
NAME=$3
TAG='tag'

python3 cifar10.py \
    --seed=$SEED \
    --lr=$LR \
    --num_epochs=$NUM_EPOCHS \
    --log_interval=$LOG_INTERVAL \
    --batch_size=$BATCH_SIZE \
    --epsilon=$EPSILON \
    --alpha=$ALPHA \
    --patience=$PATIENCE \
    --use_adv_train=$USE_ADV_TRAIN \
    --name=$NAME \
    --tag=$TAG \
    --drop_p=$DROP_P
