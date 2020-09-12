export CUDA_VISIBLE_DEVICES=$1
LR=0.001
NUM_EPOCHS=2
LOG_INTERVAL=1000
BATCH_SIZE=64
EPSILON=0.15
ALPHA=0.5
DROP_P=0.0
PATIENCE=20
ADV_PATIENCE=40
ADV_TRAIN=1

NAME='testing'
TAG='tag'

IS_DNN=0
ADV_TEST_PATH='/home/chl/dropAdv/data/''cnn-adv_train'$ADV_TRAIN'-eps'$EPSILON'.pth'
LOAD_ADV_TEST=0

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
    --adv_train=$ADV_TRAIN \
    --adv_test_path=$ADV_TEST_PATH \
    --load_adv_test=$LOAD_ADV_TEST
