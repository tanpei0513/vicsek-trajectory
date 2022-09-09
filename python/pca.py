from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import zscore

import numpy as np
from itertools import permutations, groupby
import imageio
import shutil
import glob
import os



def permute_set(iterable):
    # permute_set([1,2,3]) --> [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    return [list(p) for p in permutations(set(np.int_(iterable)))]


def set_freq(iterable):
    return [len(list(group)) for key, group in groupby(sorted(iterable))]


def find_best_accu(type_orig, type_cluster):
    all_comb = permute_set(type_cluster)
    freq = set_freq(type_orig)
    best_type = type_orig
    best_accu = accuracy_score(type_orig, type_cluster)

    for i in range(1, len(all_comb)):
        type = np.repeat(all_comb[i], freq)
        accu = accuracy_score(type, type_cluster)
        if accu > best_accu:
            best_accu = accu
            best_type = type
        else:
            pass

    return [best_type, best_accu]


def kmeans_pca(theta_mat, type_label, time_series, pca_max_iter=50, plot_pca=False, folder="pic"):
    """For KMEANS PCA"""
    accu_array = np.empty(shape=(1, len(time_series))).squeeze()
    accu_array[:] = np.nan

    if plot_pca:

        plt.rcParams.update({'figure.max_open_warning': 0})

        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    for i in range(0, len(time_series)):
        data = theta_mat[:, 0:int(time_series[i])]
        # pca_data = np.concatenate((np.sin(data), np.cos(data)), axis=1)
        pca_data_sd = StandardScaler().fit_transform(data)
        # pca_data_sd = theta_mat[:, 0:int(time_series[i])]
        components = PCA(n_components=2).fit_transform(pca_data_sd)

        model = KMeans(n_clusters=2).fit(components)
        pca_kmeans_label = model.labels_

        [type_label2, accu_array[i]] = find_best_accu(type_label, pca_kmeans_label)

        print("Clustering ProgressDialog: ", time_series[i], "/", time_series[(len(time_series) - 1)], end="\r")

        if plot_pca:

            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].scatter(components[:, 0], components[:, 1], c=type_label2, cmap='tab20')
            ax[1].scatter(components[:, 0], components[:, 1], c=pca_kmeans_label, cmap='tab10')
            ax[2].plot(time_series, accu_array, 'm-')
            ax[0].set(xlabel='PCA 1', ylabel='PCA 2', title='Original Data Set')
            ax[1].set(xlim=ax[0].get_xlim(), ylim=ax[0].get_ylim(), xlabel='PCA 1', ylabel='PCA 2',
                      title='K-mean Cluster')
            ax[2].set(ylim=(0.5, 1), xlabel='Time point', ylabel='Accuracy', title='K-mean Cluster Accuracy')

            plt.savefig(folder + "/file%02d.png" % i)

    if plot_pca:
        writer = imageio.get_writer(folder + '/test.mp4', fps=20)
        for file in glob.glob(folder + '/*.png'):
            im = imageio.v2.imread(file)
            writer.append_data(im)
        writer.close()

    return accu_array
