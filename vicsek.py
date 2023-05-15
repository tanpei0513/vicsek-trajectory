import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering

from sklearn.metrics import silhouette_score
from utils import accu_type_score

import warnings
warnings.filterwarnings("ignore")


class Particle:
    type_num = 0
    label_list = []
    noise_list = []
    radius_list = []
    num_list = []
    vel_list = []

    def __init__(self, label, noise, radius, numbers, velocity=0.01):
        Particle.type_num += 1
        Particle.label_list.append(label)
        Particle.noise_list.append(noise)
        Particle.radius_list.append(radius)
        Particle.num_list.append(numbers)
        Particle.vel_list.append(velocity)

    def reset(self):
        Particle.type_num = 0
        Particle.label_list = []
        Particle.noise_list = []
        Particle.radius_list = []
        Particle.num_list = []
        Particle.vel_list = []


class VicsekGenerator(Particle):

    def __init__(self, label, noise, radius, numbers, velocity=0.01):
        super().__init__(label, noise, radius, numbers, velocity)

        self.type_label = None
        self.time_step = None
        self.dt = None
        self.bdy = None
        self.nclust = None
        self.accuracy_arry = None
        self.cluster_list = None
        self.cluster_method = None
        self.pca_comp = None
        self.time_series = None
        self.tmax = None
        self.ypos = None
        self.xpos = None
        self.theta = None

        self.label_list = Particle.label_list
        self.type_num = Particle.type_num
        self.noise_list = Particle.noise_list
        self.radius_list = Particle.radius_list
        self.num_list = Particle.num_list
        self.vel_list = Particle.vel_list

    def initiate(self, tmax, bdy=1, dt=1):

        self.tmax = tmax
        self.bdy = bdy
        self.dt = dt
        self.time_step = int(self.tmax / self.dt)
        self.type_label = np.repeat([*range(self.type_num)], self.num_list)
        self.simulate()

    def kdtree_md(self, duration, theta0, xpos0, ypos0):
        """
        this function is using kdtree method to model vicsek motion.

        Parameter:
        theta0 : initial theta array
        xpos0: initial x pos array
        ypos0: initial y pos array
        duration: control time-step-length of simulation.

        output:
        matrix of theta, xpos, ypos after vicsek model. Each column represents one time step
        """
        theta_mat = np.zeros(shape=(sum(self.num_list), duration))
        xpos_mat = np.zeros(shape=(sum(self.num_list), duration))
        ypos_mat = np.zeros(shape=(sum(self.num_list), duration))

        theta_mat[:, 0] = theta0
        xpos_mat[:, 0] = xpos0
        ypos_mat[:, 0] = ypos0

        for t in range(1, duration):
            xy_t = np.array(list(zip(xpos_mat[:, t - 1], ypos_mat[:, t - 1])))
            ang_t = theta_mat[:, t - 1]

            # build KD-tree.
            tree = cKDTree(xy_t, boxsize=[self.bdy, self.bdy])
            for i in range(self.type_num):
                # Separate different ptcl.
                type_idx = np.array(np.where(self.type_label == i)).squeeze()
                # neighbor search gives sparse matrix (if ptcl crash or not)
                crash_idx_mat = tree.sparse_distance_matrix(tree, max_distance=self.radius_list[i],
                                                            output_type='coo_matrix')
                # to avoid angle overlap, convert angle to complex form : e^(i*theta)
                ang_cmpx = np.exp(ang_t[crash_idx_mat.col] * 1j)
                # make complex neighbor matrix
                crashed_ang_mat = sparse.coo_matrix((ang_cmpx, (crash_idx_mat.row, crash_idx_mat.col)),
                                                    shape=crash_idx_mat.get_shape())
                # average complex number
                ang_crashed = np.squeeze(np.asarray(crashed_ang_mat.tocsr().mean(axis=1)))
                ang_t[type_idx] = np.angle(ang_crashed[type_idx])
                # add environment noise
                ang_t[type_idx] += self.noise_list[i] * np.random.uniform(-np.pi, np.pi, size=self.num_list[i])
                # save new position
                cos, sin = np.cos(ang_t[type_idx]), np.sin(ang_t[type_idx])
                xy_t[type_idx, 0] += cos * self.vel_list[i]
                xy_t[type_idx, 1] += sin * self.vel_list[i]

            # periodic boundary
            xy_t[xy_t > self.bdy] -= self.bdy
            xy_t[xy_t < 0] += self.bdy

            theta_mat[:, t] = ang_t[:]
            xpos_mat[:, t] = xy_t[:, 0]
            ypos_mat[:, t] = xy_t[:, 1]

            if t % 1000 == 0:
                # print(f"Vicsek Simulation: {t} / {self.time_series[-1]}", end="\r")
                print(f"Vicsek Simulation: {t} / {int(self.tmax)}")

        print(f"Vicsek Simulation: finish!")

        return theta_mat, xpos_mat, ypos_mat

    def simulate(self):
        # create random ptcl (Initial position: x,y; Initial direction: theta)
        num_sum = sum(self.num_list)
        theta0 = np.random.uniform(-np.pi, np.pi, size=num_sum)
        xpos0 = np.random.uniform(0, self.bdy, size=num_sum)
        ypos0 = np.random.uniform(0, self.bdy, size=num_sum)

        # get simulation result
        self.theta, self.xpos, self.ypos = self.kdtree_md(duration=self.time_step, theta0=theta0, xpos0=xpos0,
                                                          ypos0=ypos0)

    def pca(self, time_series, pca_n=2):
        """
        pca for location trajectory

        Parameter:
        time_series: indexes of time points used for pca.
        pca_n: number of pca component, default 2.

        output:
        self.pca_comp[t][n:pca_n]: pca components at time point t, n is ptcl number.
        """

        self.time_series = [int(i) for i in time_series]
        self.pca_comp = [None] * len(self.time_series)

        for i, t in enumerate(self.time_series):
            data = self.theta[:, 0:int(t)]
            # data = np.arctan2(np.sin(data), np.cos(data))
            data = np.concatenate((np.sin(data), np.cos(data)), axis=1)
            self.pca_comp[i] = PCA(n_components=pca_n, random_state=31415).fit_transform(data)
            if t % 1000 == 0:
                # print(f"PCA: {t} / {self.time_series[-1]}", end="\r")
                print(f"PCA: {t} / {self.time_series[-1]}")

        print("PCA: finish!")


    def cluster(self, cluster_method="kmeans", pca_n=2):
        """
        cluster the pca result. Occupied cluster method: kmeans, spectral.
        Give after-cluster index and accuracy matching the original type.

        Parameter:
        cluster_method: choose the cluster method. Optional: "kmeans", "spectral".
        pca_n: number of pca component, default 2.

        output:
        Return the last accuracy.
        self.cluster_list[t][n]: cluster result (an array of type-index with ptcl number n) at time point t.
        self.accuracy_arry[t]: accuracy comparing to the original type.
        """

        if self.pca_comp is None:
            raise TypeError("Need run PCA first!")

        self.cluster_method = cluster_method
        self.cluster_list = [None] * len(self.time_series)
        self.accuracy_arry = [None] * len(self.time_series)

        # clustering by selected time point (self.time_series)
        for i, t in enumerate(self.time_series):
            components = self.pca_comp[i][:, 0:pca_n]
            if cluster_method == "kmeans":
                cluster_label = KMeans(n_clusters=self.type_num, random_state=31415).fit(components).labels_
                [self.cluster_list[i], self.accuracy_arry[i]] = accu_type_score(self.type_label, cluster_label)
            elif cluster_method == "spectral":
                cluster_label = SpectralClustering(n_clusters=self.type_num, affinity='nearest_neighbors',
                                                   random_state=31415).fit(components).labels_
                [self.cluster_list[i], self.accuracy_arry[i]] = accu_type_score(self.type_label, cluster_label)
            else:
                print('Optional cluster method: "kmeans", "spectral".')
            if t % 1000 == 0:
                # print(f"cluster: {t} / {self.time_series[-1]}", end="\r")
                print(f"cluster: {t} / {self.time_series[-1]}")

        print(f"cluster: finish!")

        return self.accuracy_arry[-1]

    def simulate_repl(self, tmax_repl):
        """
        replenish the simulation. Start from the last state of former simulation.

        Parameter:
        tmax_repl = time max after replenish.

        output:
        matrix of theta, xpos, ypos after vicsek model. Each column represents one time step
        """

        dur = int(tmax_repl - self.tmax)
        theta0 = self.theta[:, -1]
        xpos0 = self.xpos[:, -1]
        ypos0 = self.ypos[:, -1]

        self.tmax = tmax_repl
        self.time_step = int(self.tmax / self.dt)
        # use last t state to generate new simulation.
        theta_add, xpos_add, ypos_add = self.kdtree_md(duration=dur, theta0=theta0, xpos0=xpos0, ypos0=ypos0)
        # combine old and new simulation result.
        self.theta = np.concatenate([self.theta, theta_add], axis=1)
        self.xpos = np.concatenate([self.xpos, xpos_add], axis=1)
        self.ypos = np.concatenate([self.ypos, ypos_add], axis=1)

    def find_reach_time(self, limit=1e6):
        """
        find the time of reaching 1.0 accuracy.
        If not reach, replenish 2-fold or 10-fold simulation time length.

        Parameter:
        limit: time limit of searching. In case of spill.

        output:
        self.accu1_time: the time of reaching perfect clustering.
        """

        if self.tmax < limit:
            self.pca(time_series=[self.time_step])
            accu = self.cluster()
            while accu < 1:
                if accu < 0.8:
                    self.simulate_repl(tmax_repl=10 * self.tmax)
                else:
                    self.simulate_repl(tmax_repl=2 * self.tmax)
                self.pca(time_series=[self.time_step])
                accu = self.cluster()
            self.accu1_time = self.binary_search(low=0, high=self.time_step) + 1
        else:
            self.accu1_time = limit

    def binary_search(self, low, high):
        """
        Binary search the closest time point that reach accuracy of 1.0.

        Parameter:
        low: left value of dichotomy.
        high: right value of dichotomy.
        """

        if high > low:
            mid = low + (high - low) // 2
            self.pca(time_series=[int(mid)])
            accu = self.cluster()
            if mid == int(mid / 2):
                return int(mid)
            elif accu < 1:  # if accuracy < 1, find right.
                return self.binary_search(low=mid + 1, high=high)
            else:
                return self.binary_search(low=low, high=mid - 1)
        else:
            return int(high)

    def ncluster(self, ncluster, pca_n=2):
        """
        silhouette analysis with n cluster.

        Parameter:
        ncluster: a list of cluster numbers.
        pca_n: number of pca component, default 2.

        Output:
        self.nclust[ncluster]: a list of tested cluster numbers.
        self.nclust_sil[t][ncluster]: silhouette scores.
        self.nclust_list[t][n]: cluster result decided by highest silhouette.
        """

        self.nclust = ncluster
        self.nclust_type = [None] * len(self.time_series)
        self.nclust_sil = [None] * len(self.time_series)
        nclust_len = len(self.nclust)

        for i, t in enumerate(self.time_series):
            # initial empty lists
            clusters = [None] * nclust_len
            sil_coef = [None] * nclust_len
            # get pca component
            pca_comp = self.pca_comp[i][:, 0:pca_n]
            for j, n in enumerate(list(self.nclust)):
                kmeans_md = KMeans(n_clusters=n, init='k-means++', random_state=31415)
                clusters[j] = kmeans_md.fit_predict(pca_comp)
                # get mean silhouette coefficient.
                sil_coef[j] = silhouette_score(pca_comp, clusters[j])
            self.nclust_type[i] = clusters[np.argmax(sil_coef)]
            self.nclust_sil[i] = sil_coef
            if t % 1000 == 0:
                # print(f"ncluster: {t} / {self.time_series[-1]}", end="\r")
                print(f"ncluster: {t} / {self.time_series[-1]}")

        print(f"ncluster: finish!")

    def order_param(self):
        """
        Vicsek model order param calculation.
        Valued by the average of velocity.

        """
        data = np.cos(self.theta)
        order = [None] * (self.type_num + 1)

        for n in range(self.type_num):
            order_t = [None] * len(self.time_series)
            idx = np.where(self.type_label == n)
            for i, t in enumerate(self.time_series):
                order_t[i] = abs(np.mean(np.mean(data[idx, 0:int(t)], axis=1)))
            order[n] = order_t
        order[-1] = self.accuracy_arry

        self.order_param = order[0:self.type_num]

    def remove_item(self, item="xpos"):
        """
        To save the storage, position matrix can be removed after simulation.
        And velocity matrix can be removed after pca.

        """
        item_list = ["xpos", "ypos", "theta"]
        if item not in item_list:
            raise ValueError(f"Invalid item name. Only capable for item in {item_list}")
        try:
            if hasattr(self, str(f"{item}")):
                if item == "xpos":
                    del [self.xpos]
                if item == "ypos":
                    del [self.ypos]
                if item == "theta":
                    del [self.theta]
        except NameError:
            pass
