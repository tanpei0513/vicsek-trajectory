import os
import numpy as np
from itertools import permutations, combinations

def combinate(cluster_label, num):
    return [list(p) for p in combinations(set(np.int_(cluster_label)),num)]

def permute(labels):
    # permute([1,2,3]) --> [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    return [list(p) for p in permutations(set(np.int_(labels)))]

def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
