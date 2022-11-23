from utils import makedir

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams.update({'figure.max_open_warning': 0,
                     'font.size': 12,
                     'animation.ffmpeg_path': "D:/Project/20220706_VicsekModel/python_test/venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe"})


def scatter_plot(x, y, type_list, labels, if_movie=False, filename="test"):

    makedir("pic")

    type_num = len(np.unique(type_list))
    color = plt.cm.tab10(range(type_num))

    fig, ax = plt.subplots()
    ax.set_xlim([np.min(x) * 1.2, np.max(x) * 1.2])
    ax.set_ylim([np.min(y) * 1.2, np.max(y) * 1.2])
    ax.set_title("Time at: {}".format(0))
    ax.set(xlabel="PCA1", ylabel="PCA2")
    scat = ax.scatter(x[-1], y[-1], color=plt.cm.tab10(np.int_(type_list[-1])), s=30)
    for i in range(type_num):
        exec('scat{} = ax.scatter(None, None, color=color[{}], s=30, label=labels[{}])'.format(i, i, i))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderpad=1)
    plt.savefig(f'pic/{filename}.png')

    if if_movie:

        makedir("video")
        def update_scatter(j):
            ax.set_title("Time at: {}".format(j + 1))
            scat.set_offsets(np.c_[x[j], y[j]])
            scat.set_color(plt.cm.tab10(np.int_(type_list[j])))
            return scat,

        anim = animation.FuncAnimation(fig, update_scatter, np.arange(0, len(x)))
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')

    plt.close()


def arrow_plot(x, y, theta, bdy, type_list, labels, if_movie=True, filename="test"):

    type_num = len(np.unique(type_list))

    color = plt.cm.tab10(range(type_num))
    color_list = plt.cm.tab10(type_list)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim([0, bdy])
    ax.set_ylim([0, bdy])
    ax.set_title("")
    qv_args = dict(units='width', pivot='tail', width=0.005, headaxislength=8, headlength=8, headwidth=4)
    for i in range(type_num):
        exec('qv0 = ax.quiver(None, None, 1, 1, color=color[{}], label=labels[{}])'.format(i, i))
    qv = ax.quiver(x[:, 0], y[:, 0], np.cos(theta[:, 0]), np.sin(theta[:, 0]), color=color_list[0], **qv_args)
    # qv0 = ax.scatter(x[:, 0], y[:, 0], color="black", s=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderpad=1)
    plt.show()

    if if_movie:
        makedir("video")
        def update_arrow(i):
            ax.set_title("Time at: {}".format(i + 1))
            qv.set_offsets(np.c_[x[:, i], y[:, i]])
            qv.set_UVC(np.cos(theta[:, i]), np.sin(theta[:, i]))
            return qv,

        anim = animation.FuncAnimation(fig, update_arrow, np.arange(0, len(x[0, :])), interval=1)
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')
    plt.close()


def line_plot(x, y, ylim=[0.5,1], xlabel="time", ylabel="accuracy",if_movie=False, filename="test"):
    makedir("pic")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylim(ylim)
    color_list = plt.cm.tab10(4)

    ax.set_title("")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    line, = ax.plot(x, y, color=color_list, linewidth=3)
    fig.savefig(f"pic/{filename}.png")

    if if_movie:
        makedir("video")
        def update_line(i):
            ax.set_title(f"At {xlabel} = {x[i]}, {ylabel} = {int(y[i])}")
            line.set_xdata(x[:i])
            line.set_ydata(y[:i])
            return line,

        anim = animation.FuncAnimation(fig, update_line, np.arange(0, len(x)), interval=1)
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')


def dot_line_plot(x, y, labels, xlabel="n cluster", ylabel="normalized inertia score",if_movie=False, filename="test"):
    # makedir("pic")
    fig, ax = plt.subplots(figsize=(8, 6))
    color_list = plt.cm.tab10(4)
    ax.set_title("")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    line, = ax.plot(x[-1], y[-1], "-o", color=color_list, linewidth=3)
    plt.show()
    # fig.savefig(f"pic/{filename}.png")
    if if_movie:
        makedir("video")
        def update_line(i):
            ax.set_title(f"Time at = {labels[i]}")
            line.set_xdata(x[i])
            line.set_ydata(y[i])
            return line,

        anim = animation.FuncAnimation(fig, update_line, np.arange(0, len(labels)), interval=1)
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')


