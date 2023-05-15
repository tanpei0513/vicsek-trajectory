import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ptcl_plot(ax, x, y, theta, type_list, title=None):
    # Make color list
    type_uniq = np.unique(type_list)
    color_set = sns.color_palette('deep', 10)
    # Arrow type (triangle)
    qv_args = dict(units='width', pivot='mid', scale=18, width=0.02, headaxislength=6, headlength=6, headwidth=4)
    for idx, item in enumerate(type_uniq):
        idx_list = np.where(type_list == item)
        xx, yy, tt, cc = x[idx_list], y[idx_list], theta[idx_list], color_set[item]
        qv = ax.quiver(xx, yy, np.cos(tt), np.sin(tt), color=cc, edgecolor=cc, linewidth=1,
                       alpha=0.6, **qv_args)
    ax.set(xlabel="", ylabel="", title=title)
    # Hide interval tick
    ax.set_xticks([])
    ax.set_yticks([])


def pca_plot(ax, x, y, type_list, labels=None, title=None, **kwargs):
    # make color list
    type_uniq = np.unique(type_list)
    color_set = sns.color_palette('deep', 10)
    for idx, item in enumerate(type_uniq):
        idx_list = np.where(type_list == item)
        xx, yy, cc = x[idx_list], y[idx_list], color_set[int(item)]
        if labels is not None:
            ax.scatter(xx, yy, color=cc, label=labels[idx], edgecolor=cc, alpha=0.6, s=75, **kwargs)
        else:
            ax.scatter(xx, yy, color=cc, edgecolor=cc, alpha=0.6, s=75, **kwargs)
    if labels is not None:
        # Legend with color
        lg = ax.legend(loc='upper right', labelspacing=0.25, borderaxespad=0., handletextpad=0, edgecolor='black',
                       fancybox=False, framealpha=0.8)
        for h, t in zip(lg.legendHandles, lg.get_texts()):
            t.set_color([*h.get_facecolor()[0][0:3], 1.0])
    ax.set(xlabel="PCA1", ylabel="PCA2", title=title)
    ax.set_xticks([])
    ax.set_yticks([])

def accu_plot(ax, x, y, **kwargs):
    color = sns.color_palette('deep', 4)[-1]
    ax.plot(x, y, "-o", color=color, **kwargs)
    ax.set(xlabel="Time", ylabel="Accuracy")



