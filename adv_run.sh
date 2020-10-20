# model and gpu
IS_DNN=0
export CUDA_VISIBLE_DEVICES=$1

# training param
DATASET='mnist' # mnist or cifar10
MODEL='modelB' # base, small, or large (for cifar10) || lenet, modelA, or modelB (for mnist)
LR=0.001
NUM_EPOCHS=1
BATCH_SIZE=64
EPSILON=0.3
ALPHA=0.5
DROP_P=$3
PATIENCE=20

# adversarial training
ADV_TRAIN=$2

# neptune
NAME='exp20-whitebox'
TAG='clean-train'

for i in 1 2 3
do
    LOAD_ADV_TEST=0
    ADV_TEST_OUT_PATH=$HOME'/dropAdv/data/'$DATASET'-'$MODEL'-eps'$EPSILON'-drop_p'$DROP_P'.run'$i'.ae'

    #LOAD_ADV_TEST=1
    #ADV_TEST_PATH1=$HOME'/dropAdv/data/CNN-small-eps'$EPSILON'-drop_p0.0.run1.ae'
    #ADV_TEST_PATH2=$HOME'/dropAdv/data/CNN-small-eps'$EPSILON'-drop_p0.0.run2.ae'
    #ADV_TEST_PATH3=$HOME'/dropAdv/data/CNN-small-eps'$EPSILON'-drop_p0.0.run3.ae'
    #ADV_TEST_PATH4=$HOME'/dropAdv/data/CNN-small-eps'$EPSILON'-drop_p0.0.run4.ae'
    #ADV_TEST_PATH5=$HOME'/dropAdv/data/CNN-small-eps'$EPSILON'-drop_p0.0.run5.ae'
    #ADV_TEST_PATH6=$HOME'/dropAdv/data/CNN-large-eps'$EPSILON'-drop_p0.0.run1.ae'
    #ADV_TEST_PATH7=$HOME'/dropAdv/data/CNN-large-eps'$EPSILON'-drop_p0.0.run2.ae'
    #ADV_TEST_PATH8=$HOME'/dropAdv/data/CNN-large-eps'$EPSILON'-drop_p0.0.run3.ae'
    #ADV_TEST_PATH9=$HOME'/dropAdv/data/CNN-large-eps'$EPSILON'-drop_p0.0.run4.ae'
    #ADV_TEST_PATH10=$HOME'/dropAdv/data/CNN-large-eps'$EPSILON'-drop_p0.0.run5.ae'

    python3 cifar10.py \
        --dataset=$DATASET \
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
        --adv_test_path4=$ADV_TEST_PATH4 \
        --adv_test_path5=$ADV_TEST_PATH5 \
        --adv_test_path6=$ADV_TEST_PATH6 \
        --adv_test_path7=$ADV_TEST_PATH7 \
        --adv_test_path8=$ADV_TEST_PATH8 \
        --adv_test_path9=$ADV_TEST_PATH9 \
        --adv_test_path10=$ADV_TEST_PATH10 \
        --load_adv_test=$LOAD_ADV_TEST \
        --adv_test_out_path=$ADV_TEST_OUT_PATH
done
