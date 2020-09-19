# model and gpu
IS_DNN=0
export CUDA_VISIBLE_DEVICES=$1

# training param
LR=0.001
NUM_EPOCHS=2000
BATCH_SIZE=64
EPSILON=0.15
ALPHA=0.5
DROP_P=0.25
PATIENCE=20

# adversarial training
ADV_TRAIN=$2

# neptune
NAME='dropout-0.25'
TAG='tag'

#LOAD_ADV_TEST=0
#ADV_TEST_OUT_PATH='/home/chl/dropAdv/data/''cnn-adv_train'$ADV_TRAIN'-eps'$EPSILON'.pth'

LOAD_ADV_TEST=1
ADV_TEST_PATH1='/home/chl/dropAdv/data/''cnn-adv_train0-eps0.15''.pth'
ADV_TEST_PATH2='/home/chl/dropAdv/data/''cnn-adv_train1-eps0.15''.pth'
ADV_TEST_PATH3='/home/chl/dropAdv/data/''cnn-adv_train2-eps0.15''.pth'

python3 cifar10.py \
    --lr=$LR \
    --num_epochs=$NUM_EPOCHS \
    --batch_size=$BATCH_SIZE \
    --epsilon=$EPSILON \
    --alpha=$ALPHA \
    --patience=$PATIENCE \
    --name=$NAME \
    --tag=$TAG \
    --drop_p=$DROP_P \
    --is_dnn=$IS_DNN \
    --adv_train=$ADV_TRAIN \
    --adv_test_path1=$ADV_TEST_PATH1 \
    --adv_test_path2=$ADV_TEST_PATH2 \
    --adv_test_path3=$ADV_TEST_PATH3 \
    --load_adv_test=$LOAD_ADV_TEST \
    --adv_test_out_path=$ADV_TEST_OUT_PATH
