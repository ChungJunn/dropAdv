SEED=93
LR=0.001
NUM_EPOCHS=2000
LOG_INTERVAL=1000
BATCH_SIZE=64
EPSILON=$1
ALPHA=0.5
DROP_P=0.4
PATIENCE=20
NAME=$2
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
    --name=$NAME \
    --tag=$TAG \
    --drop_p=$DROP_P
