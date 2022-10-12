import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score

from utils import permute

class particle(object):
    type_num = 0
    label_list = []
    noise_list = []
    radius_list = []
    numbers_list = []

    def __init__(self, label, noise, radius, numbers):
        # self.label = label
        # self.noise = noise
        # self.radius = radius
        # self.numbers = numbers

        particle.type_num += 1
        particle.label_list.append(label)
        particle.noise_list.append(noise)
        particle.radius_list.append(radius)
        particle.numbers_list.append(numbers)


class vicsekGenerator(particle):

    def __init__(self, label, noise, radius, numbers):
        super().__init__(label, noise, radius, numbers)

    def initiate(self, tmax, bdy=1, dt=1, vel=0.01):

        self.tmax = tmax

        self.bdy = bdy
        self.dt = dt
        self.vel = vel

        num_sum = sum(particle.numbers_list)
        self.time_step = int(self.tmax / self.dt)
        self.type_label = np.repeat([*range(particle.type_num)], particle.numbers_list)

        # parameter initialization (theta: angle, xpos: x-axis position, ypos: y-axis position)
        self.theta_matrix = np.zeros(shape=(num_sum, self.time_step))
        self.xpos_matrix = np.zeros(shape=(num_sum, self.time_step))
        self.ypos_matrix = np.zeros(shape=(num_sum, self.time_step))

        self.theta_matrix[:, 0] = np.random.uniform(-np.pi, np.pi, size=num_sum)
        self.xpos_matrix[:, 0] = np.random.uniform(0, self.bdy, size=num_sum)
        self.ypos_matrix[:, 0] = np.random.uniform(0, self.bdy, size=num_sum)

        self.simulate()

    def simulate(self):
        radi = particle.radius_list
        eta = particle.noise_list
        num = particle.numbers_list
        type_num = particle.type_num
        vels = [self.vel] * type_num

        for t in range(1, self.time_step):
            xy_at_t = np.array(list(zip(self.xpos_matrix[:, t - 1], self.ypos_matrix[:, t - 1])))
            angles_at_t = self.theta_matrix[:, t - 1]

            tree = cKDTree(xy_at_t, boxsize=[self.bdy, self.bdy])  # build KD-tree

            for i in range(type_num):  # radius and noise varies. So separately calculate.
                type_idx = np.array(np.where(self.type_label == i)).squeeze()  # select ptcl by type label.
                crash_idx_mat = tree.sparse_distance_matrix(tree, max_distance=radi[i], output_type='coo_matrix')  # radius neighbor search
                angle_complex = np.exp(angles_at_t[crash_idx_mat.col] * 1j)  # convert angle to complex number
                crashed_angle_mat = sparse.coo_matrix((angle_complex, (crash_idx_mat.row, crash_idx_mat.col)), shape=crash_idx_mat.get_shape())  # make complex neighbor matrix
                angle_crashed = np.squeeze(np.asarray(crashed_angle_mat.tocsr().mean(axis=1)))  # average complex number
                angles_at_t[type_idx] = np.angle(angle_crashed[type_idx])  # replace old angle
                angles_at_t[type_idx] += eta[i] * np.random.uniform(-np.pi, np.pi, size=num[i])  # add noise

                cos, sin = np.cos(angles_at_t[type_idx]), np.sin(angles_at_t[type_idx])

                xy_at_t[type_idx, 0] += cos * vels[i]
                xy_at_t[type_idx, 1] += sin * vels[i]

            xy_at_t[xy_at_t > self.bdy] -= self.bdy
            xy_at_t[xy_at_t < 0] += self.bdy

            self.theta_matrix[:, t] = angles_at_t[:]
            self.xpos_matrix[:, t] = xy_at_t[:, 0]
            self.ypos_matrix[:, t] = xy_at_t[:, 1]

            print("Vicsek Simulation: ", t + 1, "/", self.time_step, end="\r")

    def pca(self, time_series=None, cluster_type=None):

        if time_series==None:
            self.time_series = np.arange(1, self.time_step, self.dt)
        else:
            self.time_series = time_series

        self.pca_component = [None] * len(self.time_series)
        self.cluster_types = [None] * len(self.time_series)
        self.accuracy_arry = [None] * len(self.time_series)


        for i in range(len(self.time_series)):
            data = self.theta_matrix[:, 0:int(self.time_series[i])]
            pca_data = np.concatenate((np.sin(data), np.cos(data)), axis=1)
            pca_data_sd = StandardScaler().fit_transform(pca_data)
            components = PCA(n_components=self.type_num).fit_transform(pca_data_sd)

            if cluster_type == "kmeans":
                cluster_label = KMeans(n_clusters=self.type_num).fit(components).labels_
                [self.cluster_types[i], self.accuracy_arry[i]] = self.accuracy(cluster_label)
            elif cluster_type == "spectral":
                cluster_label = SpectralClustering(n_clusters=self.type_num, affinity ='nearest_neighbors').fit(components).labels_
                [self.cluster_types[i], self.accuracy_arry[i]] = self.accuracy(cluster_label)

            self.pca_component[i] = components
            print("PCA: ", i, "/", len(self.time_series), end="\r")

    def accuracy(self, cluster_label):
        all_comb = permute(cluster_label)
        best_type = cluster_label
        best_accu = accuracy_score(self.type_label, cluster_label)

        if best_accu > 0.8:
            return [best_type, best_accu]
        else:
            for i in range(1, len(all_comb)): # throughout every possible combinations except comb[0]
                type_cluster_switch = np.zeros(shape=[len(cluster_label), ])
                for j in range(len(all_comb[i])): # change label from comb[0] to comb[i]
                    type_cluster_switch[np.where(cluster_label == all_comb[0][j])] = all_comb[i][j]
                accu = accuracy_score(self.type_label, type_cluster_switch)
                if accu > best_accu:
                    best_accu = accu
                    best_type = type_cluster_switch
                else:
                    pass
            return [best_type, best_accu]

    def remove_theta(self):
        try:
            if hasattr(self, "theta_matrix"):
                del [self.theta_matrix]
        except NameError:
            pass

    def remove_xpos(self):
        try:
            if hasattr(self, "xpos_matrix"):
                del [self.xpos_matrix]
        except NameError:
            pass

    def remove_ypos(self):
        try:
            if hasattr(self, "ypos_matrix"):
                del [self.ypos_matrix]
        except NameError:
            pass

    def reset(self):
        self.type_num = 0
        self.label_list = []
        self.noise_list = []
        self.radius_list = []
        self.numbers_list = []

        dic = vars(self)
        for i in dic.keys():
            dic[i] = None