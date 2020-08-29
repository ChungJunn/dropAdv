SEED=93
LR=0.001
NUM_EPOCHS=100
LOG_INTERVAL=1000
BATCH_SIZE=32
EPSILON=0.1
ALPHA=0.5
OUT_FILE='pretrained.pth'
USE_DROPOUT=1
PATIENCE=5
USE_ADV_TRAIN=0
NAME='dropout+adv'
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
    --use_dropout=$USE_DROPOUT \
    --patience=$PATIENCE \
    --use_adv_train=$USE_ADV_TRAIN \
    --name=$NAME \
    --tag=$TAG