# model and gpu
IS_DNN=0
export CUDA_VISIBLE_DEVICES=$1

# training param
MODEL=$2 # base, small, or large (for CNN)
LR=0.001
NUM_EPOCHS=2000
BATCH_SIZE=64
EPSILON=0.15
ALPHA=0.5
DROP_P=0.0
PATIENCE=20

# adversarial training
ADV_TRAIN=0

# neptune
NAME='test-clean-trained-model'
TAG='none'

LOAD_ADV_TEST=0
ADV_TEST_OUT_PATH=$HOME'/dropAdv/data/'$MODEL'-eps'$EPSILON'-drop_p'$DROP_P'.ae'

#LOAD_ADV_TEST=1
#ADV_TEST_PATH1=$HOME'/dropAdv/data/cnn-adv_train0-eps0.05.ae'
#ADV_TEST_PATH2=$HOME'/dropAdv/data/cnn-adv_train0-eps0.15-drop_p0.0.ae'
#ADV_TEST_PATH3=$HOME'/dropAdv/data/cnn-adv_train0-eps0.25.ae'

python3 cifar10.py \
    --model=$MODEL \
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
