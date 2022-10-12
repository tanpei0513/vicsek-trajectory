import numpy as np

from vicsek import vicsekGenerator
from plot import plotPCA, plotArrow, plotAccuracy

#################
##### vicsek simulations
#################

np.random.seed(314)
p = vicsekGenerator(label="type1", noise=0.1, radius=0.1, numbers=100)
p = vicsekGenerator(label="type2", noise=0.3, radius=0.1, numbers=100)

p.initiate(tmax=1000)
# plotArrow(model=p)

p.pca(cluster_type="kmeans")
# plotPCA(model=p)
# plotAccuracy(model=p)
