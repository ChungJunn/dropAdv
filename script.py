import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help='', default=0)
parser.add_argument('--adv_train', type=int, help='', default=0)
args = parser.parse_args()

n_exp = 5
cmd = './adv_run.sh ' + str(args.gpu) + ' ' + str(args.adv_train)

for i in range(n_exp):
    os.system(cmd)
    
