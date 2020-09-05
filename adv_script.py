import os

'''
ps = [0.1, 0.2, 0.3, 0.4, 0.5]
n_exp = 3
prefix = './adv_run.sh '
for p in ps:
    for i in range(n_exp):
        name = 'cnn-drop_p' + str(p) + '-run-' + str(i)
        cmd = prefix + str(p) + ' ' + name
        os.system(cmd)
'''

epss = [0.25, 0.3]
n_exp = 3
prefix = './adv_run.sh '
for eps in epss:
    for i in range(n_exp):
        name = 'cnn-eps' + str(eps) + '-run-' + str(i)
        cmd = prefix + str(eps) + ' ' + name
        os.system(cmd)
