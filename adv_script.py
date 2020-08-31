import os
# drop_p, use_adv_train

drop_ps= [0]
use_adv_trains=[1]

n_exp = 5
prefix = './adv_run.sh '

for p in drop_ps:
    for use_adv_train in use_adv_trains:
        for i in range(n_exp):
            name = 'normalValidation-drop' + str(p) + '-adv_train' + str(use_adv_train) + '-run' + str(i)
            cmd = prefix + str(p) + ' ' + str(use_adv_train) + ' ' + name
            os.system(cmd)
