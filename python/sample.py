#################
##### vicsek simulation
#################
import numpy as np
from vicsek import VicsekGenerator

np.random.seed(0)

p = VicsekGenerator(label="type1", noise=0.1, radius=0.05, numbers=200)
p = VicsekGenerator(label="type2", noise=0.3, radius=0.05, numbers=200)
p = VicsekGenerator(label="type3", noise=0.5, radius=0.05, numbers=200)
p.initiate(tmax=1e3)
p.reach_time()
p.time_accu1

t = [*np.arange(2, 100), *np.arange(100, p.tmax+100, 100)]
p.pca(time_series=t, plot=True)
p.cluster(cluster_type="kmeans",plot=True)
p.ncluster(ncluster=[*range(2, 10 + 1)], plot=True)

p.ptcl_plot(time_interval=[*np.arange(100)])
p.order_param(plot=True)

p.remove_item("xpos")
p.remove_item("ypos")

p.reset()



#################
##### Dorsogna simulation
#################
from dorsogna import DorsognaGenerator

np.random.seed(0)
p = DorsognaGenerator(alpha=1.5,beta=0.80,cA=1.0,cR=0.9,lA=1.0,lR=0.9,numbers=100,label="type1")
p = DorsognaGenerator(alpha=1.5,beta=0.775,cA=1.0,cR=0.9, lA=1.0,lR=0.9,numbers=100,label="type2")
p.initiate(tmax=2e5)
p.ptcl_plot(time_interval=[*np.arange(50)])

t = [*np.arange(2, 100),
     *np.arange(100, 1e3, 10),
     *np.arange(1e3, 1e4, 100),
     *np.arange(1e4, np.ceil((p.tmax) / 500) * 500, 500)]
p.pca(time_series=t,plot=True)
p.cluster(cluster_type="spectral",plot=True)
p.reset()