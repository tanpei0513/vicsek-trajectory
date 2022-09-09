import os
import pickle
import random
import time
import numpy as np
from vicsek_python import vicsek_kdtree
from pca import kmeans_pca

time_start = time.time()

size_bdy = 5
num = [50, 50]
radi = [0.1, 0.1]
vel = [0.1, 0.1]
eta = [0.1, 0.2]
deltat = 0.1
tmax = 1e3

repeat_times = 50
time_series = np.linspace(tmax / deltat / 50, tmax / deltat, num=50)

# seq = np.linspace(0.001, 0.01, num=10)  # radius seq
# seq = np.concatenate((np.linspace(0.01, 0.1, num=10), np.linspace(0.2, 0.5, num=4)), axis=0)  # noise seq
seq = np.concatenate((np.linspace(20, 100, num=5), np.linspace(200, 1000, num=9)), axis=0) # num seq


dirname = 'pklfiles/' + 'num2'
if not os.path.exists(dirname):
    os.makedirs(dirname)

for i in seq:
    # radi = [0.10, i]
    # eta = [0.10, i]
    num = [int(round(i/2)), int(round(i/2))]

    accu_matrix = np.zeros(shape=(len(time_series), repeat_times))
    time_start = time.time()

    for rand in range(0, repeat_times):
        random.seed(rand + 31415)
        print("repeat", rand + 1, "/", repeat_times, " Start!")
        [theta_mat, type_label] = vicsek_kdtree(size_bdy=size_bdy, num=num, radi=radi, vel=vel, eta=eta, deltat=deltat,
                                                tmax=tmax)
        accu_matrix[:, rand] = kmeans_pca(theta_mat=theta_mat, type_label=type_label,
                                          time_series=time_series)

    file_name = 'N{0}_R{1}_V{2}_S{3}_dt{4}_tm{5}_bdy{6}'.format(''.join(list(map(str, num))), ''.join(list(map(str, radi))),
                                                         ''.join(list(map(str, vel))), ''.join(list(map(str, eta))),
                                                         str(deltat), str(round(tmax)), str(round(size_bdy)))
    file_name = file_name.replace('.', '') + '_accu.pkl'
    f = open(dirname + '/' + file_name, 'wb')
    pickle.dump([accu_matrix, time_series, seq], f)
    f.close()

time_end = time.time()
print("Total time is", time_end - time_start)

