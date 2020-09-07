import os
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help='', default=0)
parser.add_argument('--drop_p', type=float, help='', default=0)
parser.add_argument('--adv_train', type=int, help='', default=0)
args = parser.parse_args()

n_exp = 3

prefix = './adv_run2.sh '
for i in range(n_exp):
    name = 'cnn-advtr' + str(args.adv_train) + '-drop' + str(args.drop_p) + '-run' + str(i)
    cmd = prefix + str(args.gpu) + ' ' + str(args.drop_p) + ' ' + str(args.adv_train) + ' ' + name
    os.system(cmd)

