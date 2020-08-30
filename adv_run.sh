SEED=93
LR=0.001
NUM_EPOCHS=1000
LOG_INTERVAL=1000
BATCH_SIZE=64
EPSILON=0.1
ALPHA=0.5
OUT_FILE='pretrained.pth'
DROP_P=0.0
PATIENCE=20
USE_ADV_TRAIN=0
NAME='DNN-testing'
TAG='tag'

python3 cifar10.py \
    --seed=$SEED \
    --lr=$LR \
    --num_epochs=$NUM_EPOCHS \
    --log_interval=$LOG_INTERVAL \
    --batch_size=$BATCH_SIZE \
    --epsilon=$EPSILON \
    --alpha=$ALPHA \
    --out_file=$OUT_FILE \
    --patience=$PATIENCE \
    --use_adv_train=$USE_ADV_TRAIN \
    --name=$NAME \
    --tag=$TAG \
    --drop_p=$DROP_P
