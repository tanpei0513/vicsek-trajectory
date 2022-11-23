import numpy as np

from vicsek_1122 import VicsekGenerator


#################
##### vicsek simulations
#################

np.random.seed(1)

p = VicsekGenerator(label="type1", noise=0.1, radius=0.1, numbers=100)
p = VicsekGenerator(label="type2", noise=0.3, radius=0.1, numbers=100)
# p = VicsekGenerator(label="type3", noise=0.5, radius=0.1, numbers=100)
p.initiate(tmax=1e3)
p.reach_time()
p.pca(time_series=[*np.arange(10, 100), *np.arange(100, np.ceil((p.tmax) / 50) * 50, 50)])
p.ncluster_plot(ncluster_num=10, if_movie=True, filename="nclust_double")
p.pca_plot(if_movie=True, filename="pca_triple")
p.pca_org_plot(if_movie=True, filename="pca_triple_org")
p.ptcl_pos_plot(time_end=500)
p.accuracy_plot(if_movie=True, filename="accuracy_triple")

p.remove_xpos()
p.remove_ypos()

p.reset()
