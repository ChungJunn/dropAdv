import os

use_drops = [0,1]
use_adv_trains=[0,1]

n_exp = 5
prefix = './adv_run.sh '

for use_drop in use_drops:
    for use_adv_train in use_adv_trains:
        tag = 'drop' + str(use_drop) + '-adv_train' + str(use_adv_train)
        for i in range(n_exp):
            cmd = prefix + str(use_drop) + ' ' + str(use_adv_train) + ' ' + tag
            os.system(cmd)
