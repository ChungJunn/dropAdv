BASE='/home/mi-lab02/0806_cnsm_dnn_rcl/data'
DATA='cnsm_exp2_1_data_total'

DATA_FILE=$BASE'/'$DATA'/train.csv'
STAT_FILE=$BASE'/'$DATA'/train.csv.stat'

EPS=0.1
MODEL_FILE='./best_model.pth'

OUT_FILE=$DATA'.adv.eps'$EPS'.csv'

python makeAE.py\
    --data_file=$DATA_FILE\
    --epsilon=$EPS\
    --stat_file=$STAT_FILE\
    --model_file=$MODEL_FILE\
    --out_file=$OUT_FILE
