import os
import numpy as np
from itertools import permutations, combinations
from sklearn.metrics import accuracy_score

def combinate(cluster_label, num):
    return [list(p) for p in combinations(set(np.int_(cluster_label)), num)]


def permute(labels):
    # permute([1,2,3]) --> [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    return [list(p) for p in permutations(set(np.int_(labels)))]


def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def accu_type_score(set1, set2):

    all_comb = permute(set2)
    l_len = len(set1)
    best_accu = 0

    for comb in all_comb:  # throughout every possible combinations except comb[0]
        type_switch = np.zeros(shape=[l_len, ])
        for idx, val in enumerate(comb):  # change label from comb[0] to comb[i]
            type_switch[np.where(set2 == all_comb[0][idx])] = val
        accu = accuracy_score(set1, type_switch)
        if accu > best_accu:
            best_accu = accu
            best_type = type_switch
            if accu > 0.8:
                break
            else:
                pass
        else:
            pass

    return [best_type, best_accu]

