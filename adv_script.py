import os
import random

ps = [0.4]
is_dnn = 0 
n_exp = 1
name = 'cnn-large-drop_p' + str(ps[0]) + '-run-3'

prefix = './adv_run.sh '
for p in ps:
    for i in range(n_exp):
        seed = random.randint(0, 100)
        #name = 'cnn-large-drop_p' + str(p) + '-run-' + str(i)
        cmd = prefix + str(is_dnn) + ' ' + str(p) + ' ' + name + ' ' + str(seed)
        os.system(cmd)

'''
epss = [0.25]
n_exp = 3
prefix = './adv_run.sh '
for eps in epss:
    for i in range(n_exp):
        name = 'cnn-eps' + str(eps) + '-run-' + str(i)
        cmd = prefix + str(eps) + ' ' + name
        os.system(cmd)
'''
