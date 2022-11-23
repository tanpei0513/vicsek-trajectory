import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import hdbscan

from sklearn.metrics import accuracy_score
from utils import permute, normalize
from plot_1122 import scatter_plot, arrow_plot, line_plot, dot_line_plot


class Particle:
    type_num = 0
    label_list = []
    noise_list = []
    radius_list = []
    numbers_list = []
    velocity_list = []

    def __init__(self, label, noise, radius, numbers, velocity=0.01):
        # self.label = label
        # self.noise = noise
        # self.radius = radius
        # self.numbers = numbers

        Particle.type_num += 1
        Particle.label_list.append(label)
        Particle.noise_list.append(noise)
        Particle.radius_list.append(radius)
        Particle.numbers_list.append(numbers)
        Particle.velocity_list.append(velocity)

    def reset(self):
        Particle.type_num = 0
        Particle.label_list = []
        Particle.noise_list = []
        Particle.radius_list = []
        Particle.numbers_list = []
        Particle.velocity_list = []


class VicsekGenerator(Particle):

    def __init__(self, label, noise, radius, numbers, velocity=0.01):
        super().__init__(label, noise, radius, numbers, velocity)

        self.label_list = Particle.label_list
        self.type_num = Particle.type_num
        self.noise_list = Particle.noise_list
        self.radius_list = Particle.radius_list
        self.numbers_list = Particle.numbers_list
        self.velocity_list = Particle.velocity_list

    def initiate(self, tmax, bdy=1, dt=1):

        self.tmax = tmax
        self.bdy = bdy
        self.dt = dt

        self.time_step = int(self.tmax / self.dt)
        self.type_label = np.repeat([*range(self.type_num)], self.numbers_list)

        self.simulate()

    def kdtree(self, tmax, theta0, xpos0, ypos0):

        theta_mat = np.zeros(shape=(sum(self.numbers_list), self.time_step))
        xpos_mat = np.zeros(shape=(sum(self.numbers_list), self.time_step))
        ypos_mat = np.zeros(shape=(sum(self.numbers_list), self.time_step))

        theta_mat[:, 0] = theta0
        xpos_mat[:, 0] = xpos0
        ypos_mat[:, 0] = ypos0

        for t in range(1, tmax):
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
                                                                                size=self.numbers_list[i])  # add noise

                cos, sin = np.cos(angles_at_t[type_idx]), np.sin(angles_at_t[type_idx])

                xy_at_t[type_idx, 0] += cos * self.velocity_list[i]
                xy_at_t[type_idx, 1] += sin * self.velocity_list[i]

            xy_at_t[xy_at_t > self.bdy] -= self.bdy
            xy_at_t[xy_at_t < 0] += self.bdy

            theta_mat[:, t] = angles_at_t[:]
            xpos_mat[:, t] = xy_at_t[:, 0]
            ypos_mat[:, t] = xy_at_t[:, 1]

            print("Vicsek Simulation: ", t + 1, "/", tmax, end="\r")

        return theta_mat, xpos_mat, ypos_mat

    def simulate(self):
        # radi = Particle.radius_list
        # eta = Particle.noise_list
        # num = Particle.numbers_list
        # type_num = Particle.type_num
        # vel = Particle.velocity_list

        num_sum = sum(self.numbers_list)

        theta0 = np.random.uniform(-np.pi, np.pi, size=num_sum)
        xpos0 = np.random.uniform(0, self.bdy, size=num_sum)
        ypos0 = np.random.uniform(0, self.bdy, size=num_sum)

        self.theta_mat, self.xpos_mat, self.ypos_mat = self.kdtree(tmax=self.time_step, theta0=theta0, xpos0=xpos0,
                                                                   ypos0=ypos0)

    def pca(self, time_series=None):

        if time_series is None:
            time_series = [*np.arange(1, 100), *np.arange(100, np.ceil((self.tmax) / 20) * 20, 20)]
            self.time_series = [int(i) for i in time_series]
        else:
            try:
                self.time_series = [int(i) for i in time_series]
            except ValueError:
                pass

        self.pca_component = [None] * len(self.time_series)

        for i, t in enumerate(self.time_series):
            data = self.theta_mat[:, 0:int(t)]
            data = np.concatenate((np.sin(data), np.cos(data)), axis=1)
            self.pca_component[i] = PCA(random_state=31415).fit_transform(data)
            print(f"pca: {t} / {self.time_series[-1]}")

    def cluster(self, pca_n=2, cluster_type="kmeans", eps=0.3, min_sample=10):

        # self.pca_component = [None] * len(self.time_series)
        self.cluster_types = [None] * len(self.time_series)
        self.accuracy_arry = [None] * len(self.time_series)

        for i in range(len(self.time_series)):
            components = self.pca_component[i][:, 0:pca_n]
            if cluster_type == "kmeans":
                cluster_label = KMeans(n_clusters=self.type_num).fit(components).labels_
                [self.cluster_types[i], self.accuracy_arry[i]] = self.accuracy(cluster_label)
            elif cluster_type == "spectral":
                cluster_label = SpectralClustering(n_clusters=self.type_num, affinity='nearest_neighbors').fit(
                    components).labels_
                [self.cluster_types[i], self.accuracy_arry[i]] = self.accuracy(cluster_label)
            elif cluster_type == 'dbscan':
                cluster_label = DBSCAN(eps=eps, min_samples=min_sample).fit(components).labels_
                self.cluster_types[i] = cluster_label
                # [self.cluster_types[i], self.accuracy_arry[i]] = self.accuracy(cluster_label)
            elif cluster_type == 'hdbscan':
                cluster_label = hdbscan.HDBSCAN().fit(components).labels_
                self.cluster_types[i] = cluster_label
                # [self.cluster_types[i], self.accuracy_arry[i]] = self.accuracy(cluster_label)
            elif cluster_type == 'optics':
                cluster_label = OPTICS(min_samples=4).fit(components).labels_
                self.cluster_types[i] = cluster_label
                # [self.cluster_types[i], self.accuracy_arry[i]] = self.accuracy(cluster_label)
            elif cluster_type == "hierarchy":
                hierarchical_cluster = AgglomerativeClustering(n_clusters=self.type_num, affinity='euclidean',
                                                               linkage='ward')
                cluster_label = hierarchical_cluster.fit_predict(components)
                [self.cluster_types[i], self.accuracy_arry[i]] = self.accuracy(cluster_label)
            else:
                print("Invalid cluster type!")

    def time_expand(self, tmax_new):

        time_expand = int(tmax_new - self.tmax)

        self.tmax = tmax_new
        self.time_step = int(self.tmax / self.dt)

        theta0 = self.theta_mat[:, -1]
        xpos0 = self.xpos_mat[:, -1]
        ypos0 = self.ypos_mat[:, -1]

        theta_mat2, xpos_mat2, ypos_mat2 = self.kdtree(tmax=time_expand, theta0=theta0, xpos0=xpos0, ypos0=ypos0)

        self.theta_mat = np.concatenate([self.theta_mat, theta_mat2], axis=1)
        self.xpos_mat = np.concatenate([self.xpos_mat, xpos_mat2], axis=1)
        self.ypos_mat = np.concatenate([self.ypos_mat, ypos_mat2], axis=1)

    def accuracy(self, cluster_label):
        # comb = combinate(cluster_label, self.type_num)
        all_comb = permute(cluster_label)
        best_type = cluster_label
        best_accu = accuracy_score(self.type_label, cluster_label)

        if best_accu > 0.75:
            return [best_type, best_accu]
        else:
            for i in range(1, len(all_comb)):  # throughout every possible combinations except comb[0]
                type_cluster_switch = np.zeros(shape=[len(cluster_label), ])
                for j in range(len(all_comb[i])):  # change label from comb[0] to comb[i]
                    type_cluster_switch[np.where(cluster_label == all_comb[0][j])] = all_comb[i][j]
                accu = accuracy_score(self.type_label, type_cluster_switch)
                if accu > best_accu:
                    best_accu = accu
                    best_type = type_cluster_switch
                else:
                    pass
            return [best_type, best_accu]

    def reach_time(self, pca_num=2, cluster_type="kmeans", limit=1e6):

        self.time_accu1 = 0

        if self.tmax < limit:
            self.pca(time_series=[self.time_step])
            self.cluster(pca_n=pca_num, cluster_type=cluster_type)
            while self.accuracy_arry[0] < 1:
                if self.accuracy_arry[0] < 0.75:
                    self.time_expand(tmax_new=10 * self.tmax)
                else:
                    self.time_expand(tmax_new=2 * self.tmax)
                self.pca(time_series=[self.time_step])
                self.cluster(pca_n=pca_num, cluster_type=cluster_type)
        else:
            self.time_accu1 = limit

        self.time_accu1 = self.search_closest(low=0, high=self.time_step, cluster_type=cluster_type) + 1

    def search_closest(self, low, high, cluster_type="kmeans"):
        if high > low:
            mid = low + (high - low) // 2
            self.pca(time_series=[int(mid)])
            self.cluster(pca_n=2, cluster_type=cluster_type)
            accu = self.accuracy_arry[0]
            if mid == int(mid / 2):
                return int(mid)
            elif accu < 1:
                return self.search_closest(low=mid + 1, high=high, cluster_type=cluster_type)
            else:
                return self.search_closest(low=low, high=mid - 1, cluster_type=cluster_type)
        else:
            return int(high)

    def pca_plot(self, if_movie=True, filename="pca"):
        comp = self.pca_component
        pca1 = [item[:, 0] for item in comp]
        pca2 = [item[:, 1] for item in comp]
        scatter_plot(x=pca1, y=pca2, type_list=self.cluster_types, labels=self.label_list, if_movie=if_movie,
                     filename=filename)

    def pca_org_plot(self, if_movie=True, filename="pca_org"):
        comp = self.pca_component
        pca1 = [item[:, 0] for item in comp]
        pca2 = [item[:, 1] for item in comp]
        scatter_plot(x=pca1, y=pca2, type_list=[self.type_label] * len(pca1),
                     labels=self.label_list, if_movie=if_movie, filename=filename)

    def ptcl_pos_plot(self, time_end, if_movie=True, filename="particle_motion"):

        xpos = self.xpos_mat[:, 0:int(time_end)]
        ypos = self.ypos_mat[:, 0:int(time_end)]
        theta = self.theta_mat[:, 0:int(time_end)]
        type_list = [self.type_label] * int(time_end)

        arrow_plot(x=xpos, y=ypos, theta=theta, bdy=self.bdy, type_list=type_list, labels=self.label_list, if_movie=if_movie, filename=filename)

    def accuracy_plot(self, if_movie=False, filename="time_accuracy"):
        self.cluster(pca_n=2, cluster_type="kmeans")
        line_plot(x=self.time_series, y=self.accuracy_arry, if_movie=if_movie, filename=filename)

    def ncluster_plot(self, if_movie=False, ncluster_num=10, filename="time_ncluster"):

        self.nclust_score = [None] * len(self.time_series)
        nclust_list = [*range(1, ncluster_num + 1)]

        for i, t in enumerate(self.time_series):
            pca_comp = self.pca_component[i][:,0:2]
            score = [None] * len(nclust_list)
            for j, n in enumerate(nclust_list):
                clustering = KMeans(n_clusters=n, init='k-means++', random_state=31415)
                clustering.fit(pca_comp)
                score[j] = clustering.inertia_
            self.nclust_score[i] = score

        if if_movie:
            norm_score = [None] * len(self.nclust_score)
            for i in range(len(self.nclust_score)):
                norm_score[i] = normalize(self.nclust_score[i])
            dot_line_plot(x=[nclust_list] * len(self.time_series), y=norm_score,
                          xlabel = "n cluster", ylabel = "normalized inertia score",
                          labels=[int(i) for i in self.time_series], if_movie=if_movie, filename=filename)

    def remove_theta(self):
        try:
            if hasattr(self, "theta_matrix"):
                del [self.theta_mat]
        except NameError:
            pass

    def remove_xpos(self):
        try:
            if hasattr(self, "xpos_matrix"):
                del [self.xpos_mat]
        except NameError:
            pass

    def remove_ypos(self):
        try:
            if hasattr(self, "ypos_matrix"):
                del [self.ypos_mat]
        except NameError:
            pass
