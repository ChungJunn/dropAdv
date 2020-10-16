# model and gpu
IS_DNN=0
export CUDA_VISIBLE_DEVICES=$1

# training param
MODEL='base' # base, small, or large (for CNN)
LR=0.001
NUM_EPOCHS=1000
BATCH_SIZE=64
EPSILON=$2
ALPHA=0.5
DROP_P=$3
#RUN=$3 # only for training
PATIENCE=20

# adversarial training
ADV_TRAIN=1

# neptune
NAME='transferred-FGSM-ae'
TAG='none'

#LOAD_ADV_TEST=0
#ADV_TEST_OUT_PATH=$HOME'/dropAdv/data/CNN-'$MODEL'-eps'$EPSILON'-drop_p'$DROP_P'.run'$RUN'.ae'

LOAD_ADV_TEST=1
ADV_TEST_PATH1=$HOME'/dropAdv/data/CNN-base-eps'$EPSILON'-drop_p0.0.run1.ae'
ADV_TEST_PATH2=$HOME'/dropAdv/data/CNN-base-eps'$EPSILON'-drop_p0.0.run2.ae'
ADV_TEST_PATH3=$HOME'/dropAdv/data/CNN-base-eps'$EPSILON'-drop_p0.0.run3.ae'

for i in 1 2 3
do
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
done
