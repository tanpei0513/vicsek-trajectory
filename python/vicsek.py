import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering

from sklearn.metrics import silhouette_score
from utils import accu_type_score, normalize
from plot import scatter, arrow, line, dot_line, line_multi


class Particle:
    type_num = 0
    label_list = []
    noise_list = []
    radius_list = []
    num_list = []
    vel_list = []

    def __init__(self, label, noise, radius, numbers, velocity=0.01):
        # self.label = label
        # self.noise = noise
        # self.radius = radius
        # self.numbers = numbers

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

    def kdtree(self, duration, theta0, xpos0, ypos0):

        theta_mat = np.zeros(shape=(sum(self.num_list), duration))
        xpos_mat = np.zeros(shape=(sum(self.num_list), duration))
        ypos_mat = np.zeros(shape=(sum(self.num_list), duration))

        theta_mat[:, 0] = theta0
        xpos_mat[:, 0] = xpos0
        ypos_mat[:, 0] = ypos0

        for t in range(1, duration):
            xy_at_t = np.array(list(zip(xpos_mat[:, t - 1], ypos_mat[:, t - 1])))
            angles_at_t = theta_mat[:, t - 1]

            tree = cKDTree(xy_at_t, boxsize=[self.bdy, self.bdy])  # build KD-tree

            for i in range(self.type_num):  # radius and noise varies. So separately calculate.
                type_idx = np.array(np.where(self.type_label == i)).squeeze()  # select ptcl by type label.
                crash_idx_mat = tree.sparse_distance_matrix(tree, max_distance=self.radius_list[i],
                                                            output_type='coo_matrix')  # radius neighbor search
                angle_complex = np.exp(angles_at_t[crash_idx_mat.col] * 1j)  # convert angle to complex number
                crashed_angle_mat = sparse.coo_matrix((angle_complex, (crash_idx_mat.row, crash_idx_mat.col)),
                                                      shape=crash_idx_mat.get_shape())  # make complex neighbor matrix
                angle_crashed = np.squeeze(np.asarray(crashed_angle_mat.tocsr().mean(axis=1)))  # average complex number
                angles_at_t[type_idx] = np.angle(angle_crashed[type_idx])  # replace old angle
                angles_at_t[type_idx] += self.noise_list[i] * np.random.uniform(-np.pi, np.pi,
                                                                                size=self.num_list[i])  # add noise

                cos, sin = np.cos(angles_at_t[type_idx]), np.sin(angles_at_t[type_idx])

                xy_at_t[type_idx, 0] += cos * self.vel_list[i]
                xy_at_t[type_idx, 1] += sin * self.vel_list[i]

            xy_at_t[xy_at_t > self.bdy] -= self.bdy
            xy_at_t[xy_at_t < 0] += self.bdy

            theta_mat[:, t] = angles_at_t[:]
            xpos_mat[:, t] = xy_at_t[:, 0]
            ypos_mat[:, t] = xy_at_t[:, 1]

            print("Vicsek Simulation: ", t + 1, "/", duration, end="\r")

        return theta_mat, xpos_mat, ypos_mat

    def simulate(self):
        # radi = Particle.radius_list
        # eta = Particle.noise_list
        # num = Particle.numbers_list
        # type_num = Particle.type_num
        # vel = Particle.velocity_list

        num_sum = sum(self.num_list)
        theta0 = np.random.uniform(-np.pi, np.pi, size=num_sum)
        xpos0 = np.random.uniform(0, self.bdy, size=num_sum)
        ypos0 = np.random.uniform(0, self.bdy, size=num_sum)

        self.theta, self.xpos, self.ypos = self.kdtree(duration=self.time_step, theta0=theta0, xpos0=xpos0, ypos0=ypos0)

    def pca(self, plot, pca_n=2, time_series=None):

        if time_series is None:
            time_series = [*np.arange(1, 100), *np.arange(100, np.ceil((self.tmax) / 20) * 20, 20)]
            self.time_series = [int(i) for i in time_series]
        else:
            try:
                self.time_series = [int(i) for i in time_series]
            except ValueError:
                pass

        self.pca_n = pca_n
        self.pca_comp = [None] * len(self.time_series)

        for i, t in enumerate(self.time_series):
            data = self.theta[:, 0:int(t)]
            data = np.concatenate((np.sin(data), np.cos(data)), axis=1)
            self.pca_comp[i] = PCA(n_components=pca_n, random_state=31415).fit_transform(data)
            print(f"pca: {t} / {self.time_series[-1]}", end="\r")

        if plot:
            titles = ["Time: " + str(int(i)) for i in self.time_series]
            pca1 = [item[:, 0] for item in self.pca_comp]
            pca2 = [item[:, 1] for item in self.pca_comp]
            scatter(x=pca1, y=pca2, xlabel="PCA1", ylabel="PCA2", title=titles, type_list=[self.type_label] * len(pca1), labels=self.label_list, if_movie=True, filename="vicsek_pca_org")

    def cluster(self, plot, cluster_type="kmeans"):

        self.cluster_type = cluster_type
        self.cluster_list = [None] * len(self.time_series)
        self.accuracy_arry = [None] * len(self.time_series)

        for i, t in enumerate(self.time_series):
            components = self.pca_comp[i][:, 0:self.pca_n]
            if cluster_type == "kmeans":
                cluster_label = KMeans(n_clusters=self.type_num, random_state=31415).fit(components).labels_
                [self.cluster_list[i], self.accuracy_arry[i]] = accu_type_score(self.type_label, cluster_label)
            elif cluster_type == "spectral":
                cluster_label = SpectralClustering(n_clusters=self.type_num, affinity='nearest_neighbors', random_state=31415).fit(components).labels_
                [self.cluster_list[i], self.accuracy_arry[i]] = accu_type_score(self.type_label, cluster_label)
            else:
                print("Invalid cluster type!")
            print(f"cluster: {t} / {self.time_series[-1]}", end="\r")

        if plot:
            comp = self.pca_comp
            pca1 = [item[:, 0] for item in comp]
            pca2 = [item[:, 1] for item in comp]
            titles = [f"Time: {self.time_series[i]}. Accuracy = {round(self.accuracy_arry[i], 3)}" for i in range(len(self.time_series))]

            scatter(x=pca1, y=pca2, xlabel="PCA1", ylabel="PCA2",
                    type_list=self.cluster_list, labels=self.label_list,
                    if_movie=True, filename=f"vicsek_{cluster_type}_pca")
            if self.type_num == 2:
                line(x=self.time_series, y=self.accuracy_arry, xlabel="time", ylabel="accuracy", ylim=[0.5, 1],
                     title=titles, if_movie=True, filename=f"vicsek_{cluster_type}_accu")
            else:
                line(x=self.time_series, y=self.accuracy_arry, xlabel="time", ylabel="accuracy", ylim=[0, 1],
                     title=titles, if_movie=True, filename=f"vicsek_{cluster_type}_accu")

    def time_expand(self, tmax_new):

        dur = int(tmax_new - self.tmax)

        self.tmax = tmax_new
        self.time_step = int(self.tmax / self.dt)

        theta0 = self.theta[:, -1]
        xpos0 = self.xpos[:, -1]
        ypos0 = self.ypos[:, -1]

        theta_add, xpos_add, ypos_add = self.kdtree(duration=dur, theta0=theta0, xpos0=xpos0, ypos0=ypos0)

        self.theta = np.concatenate([self.theta, theta_add], axis=1)
        self.xpos = np.concatenate([self.xpos, xpos_add], axis=1)
        self.ypos = np.concatenate([self.ypos, ypos_add], axis=1)

    def reach_time(self, limit=1e6):

        self.time_accu1 = 0

        if self.tmax < limit:
            self.pca(pca_n=2, time_series=[self.time_step], plot=False)
            self.cluster(plot=False)
            while self.accuracy_arry[0] < 1:
                if self.accuracy_arry[0] < 0.75:
                    self.time_expand(tmax_new=10 * self.tmax)
                else:
                    self.time_expand(tmax_new=2 * self.tmax)
                self.pca(pca_n=2, time_series=[self.time_step], plot=False)
                self.cluster(plot=False)
        else:
            self.time_accu1 = limit

        self.time_accu1 = self.search_closest(low=0, high=self.time_step) + 1

    def search_closest(self, low, high):
        if high > low:
            mid = low + (high - low) // 2
            self.pca(pca_n=self.pca_n, time_series=[int(mid)], plot=False)
            self.cluster(cluster_type=self.cluster_type, plot=False)
            accu = self.accuracy_arry[0]
            if mid == int(mid / 2):
                return int(mid)
            elif accu < 1:
                return self.search_closest(low=mid + 1, high=high)
            else:
                return self.search_closest(low=low, high=mid - 1)
        else:
            return int(high)

    def ncluster(self, ncluster, plot):

        nclust_len = len(ncluster)
        self.nclust = ncluster
        self.nclust_score = [None] * len(self.time_series)
        self.nclust_list = [None] * len(self.time_series)
        self.nclust_sil = [None] * len(self.time_series)

        for i,t in enumerate(self.time_series):
            pca_comp = self.pca_comp[i][:, 0:self.pca_n]
            clusters = [None] * nclust_len
            score = [None] * nclust_len
            sil_avg = [None] * nclust_len
            for j, n in enumerate(self.nclust):
                kmean_model = KMeans(n_clusters=n, init='k-means++', random_state=31415)
                clusters[j] = kmean_model.fit_predict(pca_comp)
                sil_avg[j] = silhouette_score(pca_comp, clusters[j])
                score[j] = kmean_model.inertia_
            self.nclust_score[i] = score
            self.nclust_list[i] = clusters
            self.nclust_sil[i] = sil_avg
            print(f"ncluster: {t} / {self.time_series[-1]}", end="\r")

        if plot:
            titles = ["Time: " + str(int(i)) for i in self.time_series]
            dot_line(x=[self.nclust] * len(self.time_series), y=self.nclust_sil,
                     ylim=[np.min(self.nclust_sil) * 0.8, np.max(self.nclust_sil) * 1.2],
                     xlabel="n cluster", ylabel="silhouette score",
                     title=titles, if_movie=True, filename="vicsek_ncluster_silhouette")

            comp = self.pca_comp
            pca1 = [item[:, 0] for item in comp]
            pca2 = [item[:, 1] for item in comp]
            best_clust = [None] * len(pca1)
            for i in range(len(pca1)):
                best_clust[i] = self.nclust_list[i][np.argmax(self.nclust_sil[i])]
            scatter(x=pca1, y=pca2, xlabel="PCA1", ylabel="PCA2",
                    type_list=best_clust, if_movie=True, title=titles,
                    filename="vicsek_ncluster_silhouette_pca")

    def order_param(self, plot):

        data = np.cos(self.theta)
        order = [None] * (self.type_num+1)

        for n in range(self.type_num):
            order_t = [None] * len(self.time_series)
            idx = np.where(self.type_label == n)
            for i, t in enumerate(self.time_series):
                order_t[i] = abs(np.mean(np.mean(data[idx, 0:int(t)], axis=1)))
            order[n] = order_t
        order[-1] = self.accuracy_arry

        self.order_param = order[0:self.type_num]

        # for i,n in enumerate(order1) :
        #     order1[i] = n / p1.numbers_list[0]
        #
        # for i,n in enumerate(order2) :
        #     order2[i] = n / p1.numbers_list[1]
        # order_param = abs(np.mean(data2, axis=0))
        # order_param = order_param[[*p1.time_series],]

        # order1_sc = (order1 - np.min(order1)) / (np.max(order1) - np.min(order1))
        # order2_sc = (order2 - np.min(order2)) / (np.max(order2) - np.min(order2))

        if plot:
            line_multi(x=[self.time_series]*len(order), y=order, xlabel="time", ylabel="order parameter",
                       labels=[*self.label_list, "Accuracy"], if_movie=False, filename="vicsek_order_param")
        else:
            pass

    def ptcl_plot(self, time_interval):

        xpos = self.xpos[:, time_interval]
        ypos = self.ypos[:, time_interval]
        theta = self.theta[:, time_interval]
        type_list = [self.type_label] * len(time_interval)
        titles = ["Time: "+str(int(i)) for i in time_interval]

        arrow(x=xpos, y=ypos, theta=theta, bdy=self.bdy,
              title=titles, type_list=type_list, labels=self.label_list, if_movie=True, filename="vicsek_ptcl")

    def ncluster_elbow_plot(self):

        norm_score = [None] * len(self.nclust_score)
        for i in range(len(self.nclust_score)):
            norm_score[i] = normalize(self.nclust_score[i])
        dot_line(x=[self.nclust] * len(self.time_series), y=norm_score, ylim=[0, 1],
                 xlabel="n cluster", ylabel="normalized inertia score",
                 title=self.time_series, if_movie=True, filename="vicsek_ncluster_elbow")

    def remove_item(self, item="xpos"):
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
