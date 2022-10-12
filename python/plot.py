from utils import makedir

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams.update({'figure.max_open_warning': 0,
                     'font.size': 12,
                     'animation.ffmpeg_path': "D:/Project/20220706_VicsekModel/python_test/venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe"})



def plotPCA(model, *args, **kwargs):

    makedir("video")

    pca = model.pca_component

    pca1 = [item[:,0] for item in pca]
    pca2 = [item[:,1] for item in pca]

    types = np.unique(model.type_label)
    color = plt.cm.tab10(types)
    # color_list = plt.cm.tab10(type_label)

    fig, ax = plt.subplots()
    ax.set_xlim([np.min(pca1) - 5, np.max(pca1) + 5])
    ax.set_ylim([np.min(pca2) - 5, np.max(pca2) + 5])
    ax.set_title("Time at: {}".format(0))
    ax.set(xlabel="PCA1", ylabel="PCA2")

    scat = ax.scatter(pca1[100], pca2[100], color=plt.cm.tab10(np.int_(model.cluster_types[100])), s=30)

    for i in types:
        exec('scat{} = ax.scatter(None, None, color=color[{}], s=30, label=model.label_list[{}])'.format(i,i,i))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderpad=1)
    plt.show()


    def update_plotPCA(j):
        ax.set_title("Time at: {}".format(j + 1))
        scat.set_offsets(np.c_[pca1[j], pca2[j]])
        scat.set_color(plt.cm.tab10(np.int_(model.cluster_types[j])))
        return scat,

    anim = animation.FuncAnimation(fig, update_plotPCA, np.arange(0, len(pca1)))
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save('video/pca_method_movie.mp4', writer=FFwriter)
    # anim.save("movie/pca_method_movie.gif", writer='pillow')
    plt.close()

def plotArrow(model):

    makedir("video")

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim([0, model.bdy])
    ax.set_ylim([0, model.bdy])
    ax.set_title("Time at :{}".format(0))
    qv_args = dict(units='width', pivot='middle', width=0.005,
                   headaxislength=8, headlength=8, headwidth=4)

    types = np.unique(model.type_label)
    color = plt.cm.tab10(types)
    color_list = plt.cm.tab10(model.type_label)

    xpos = model.xpos_matrix
    ypos = model.ypos_matrix
    theta = model.theta_matrix

    for i in types:
        exec('qv0 = ax.quiver(None, None, 1, 1, color=color[{}], label=model.label_list[{}], **qv_args)'.format(i, i))

    qv = ax.quiver(xpos[:,0], ypos[:,0], np.cos(theta[:,0]), np.sin(theta[:,0]),
                    color=color_list, **qv_args)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderpad=1)
    plt.show()

    def update_plotArrow(i):
        ax.set_title("Time at: {}".format(i + 1))
        qv.set_offsets(np.c_[xpos[:,i], ypos[:,i]])
        qv.set_UVC(np.cos(theta[:,i]), np.sin(theta[:,i]))
        return qv,

    anim = animation.FuncAnimation(fig, update_plotArrow, np.arange(0, len(xpos[0,:])), interval=1)
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save('video/particle_motion_movie.mp4', writer=FFwriter)
    # anim.save("video/particle_motion_movie.gif", writer='pillow')
    plt.close()


def plotAccuracy(model):

    makedir("video")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylim([0.5, 1])

    color_list = plt.cm.tab10(4)

    ax.set_title("At time = {}, cluster accuracy = {}".format(model.time_series[-1]+1, model.accuracy_arry[-1]))
    ax.set(xlabel="Time", ylabel="accuracy")
    line, = ax.plot(model.time_series[:len(model.time_series)], model.accuracy_arry[:len(model.time_series)], color=color_list, linewidth=3)

    fig.savefig("accu_plot.png")
    plt.show()

    def update_plotAccu(i):
        ax.set_title("At time = {}, cluster accuracy = {}".format(i + 1, model.accuracy_arry[i]))
        line.set_xdata(model.time_series[:i])
        line.set_ydata(model.accuracy_arry[:i])
        return line,

    anim = animation.FuncAnimation(fig, update_plotAccu, np.arange(0, len(model.time_series)), interval=1)
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save('video/accuracy_movie.mp4', writer=FFwriter)
    # anim.save("video/accuracy_movie.gif", writer='pillow')
    plt.close()


