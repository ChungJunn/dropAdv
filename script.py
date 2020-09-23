import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help='', default=0)
parser.add_argument('--epsilon', type=float, help='', default=0)
parser.add_argument('--dropout_p', type=float, help='', default=0)
args = parser.parse_args()

n_exp = 4

adv_train = 0
cmd = './adv_run.sh ' + str(args.gpu) + ' ' + str(args.epsilon) + ' ' + str(args.dropout_p) + ' ' + str(adv_train)
for i in range(n_exp):
    os.system(cmd)

adv_train = 1
cmd = './adv_run.sh ' + str(args.gpu) + ' ' + str(args.epsilon) + ' ' + str(args.dropout_p) + ' ' + str(adv_train)
for i in range(n_exp):
    os.system(cmd)

