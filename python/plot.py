import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from utils import makedir

plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 12, 'animation.ffmpeg_path': "D:/Project/20220706_VicsekModel/python_test/venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe"})
plt.rcParams.update({'animation.ffmpeg_path': "D:/Project/20220706_VicsekModel/python_test/venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe"})


def scatter(x, y, type_list, xlabel="x", ylabel="y", title=None, labels=None, if_movie=True, filename="scatter"):

    makedir("pic")
    type_num = len(np.unique(type_list))
    color = plt.cm.tab10(range(type_num))
    fig, ax = plt.subplots()
    ax.set_xlim([np.min(x) - abs(0.2*np.min(x)), np.max(x) + abs(0.2*np.max(x))])
    ax.set_ylim([np.min(y) - abs(0.2*np.min(y)), np.max(y) + abs(0.2*np.max(y))])
    ax.set_title("")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    scat = ax.scatter(x[-1], y[-1], color=plt.cm.tab10(np.int_(type_list[-1])), s=30)
    if labels is not None:
        for i in range(type_num):
            exec('scat{} = ax.scatter(None, None, color=color[{}], s=30, label=labels[{}])'.format(i, i, i))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderpad=1)
    plt.savefig(f'pic/{filename}.pdf')

    if if_movie:
        makedir("video")
        def update_scatter(i):
            if title is not None:
                ax.set_title(f"{title[i]}")
            scat.set_offsets(np.c_[x[i], y[i]])
            scat.set_color(plt.cm.tab10(np.int_(type_list[i])))
            return scat,
        anim = animation.FuncAnimation(fig, update_scatter, np.arange(0, len(x)))
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')
    else:
        pass

    plt.close()


def arrow(x, y, theta, type_list, title=None, bdy=None, xlabel=" ", ylabel=" ", labels=None, if_movie=True, filename="arrow"):

    makedir("pic")
    type_num = len(np.unique(type_list))
    color = plt.cm.tab10(range(type_num))
    color_list = plt.cm.tab10(type_list)[0]
    fig, ax = plt.subplots(figsize=(16, 12))
    if bdy is not None:
        ax.set_xlim([0, bdy])
        ax.set_ylim([0, bdy])
    ax.set_title("")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    qv_args = dict(units='width', pivot='tail', width=0.005, headaxislength=8, headlength=8, headwidth=4)
    qv = ax.quiver(x[:, 0], y[:, 0], np.cos(theta[:, 0]), np.sin(theta[:, 0]), color=color_list, **qv_args)
    # qv0 = ax.scatter(x[:, 0], y[:, 0], color="black", s=1)
    if labels is not None:
        for i in range(type_num):
            exec('qv0 = ax.quiver(None, None, 1, 1, color=color[{}], label=labels[{}])'.format(i, i))
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderpad=1)
    # plt.savefig(f'pic/{filename}.png')

    if if_movie:
        makedir("video")
        def update_arrow(i):
            if title is not None:
                ax.set_title(f"{title[i]}")
            qv.set_offsets(np.c_[x[:, i], y[:, i]])
            qv.set_UVC(np.cos(theta[:, i]), np.sin(theta[:, i]))
            return qv,
        anim = animation.FuncAnimation(fig, update_arrow, np.arange(0, len(x[0, :])), interval=1)
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')
    plt.close()


def line(x, y, title=None, ylim=None, xlabel="x", ylabel="y", if_movie=False, filename="line"):

    makedir("pic")
    fig, ax = plt.subplots(figsize=(8, 6))
    if ylim is not None:
        ax.set_ylim(ylim)
    color_list = plt.cm.tab10(4)
    ax.set_title("")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    line, = ax.plot(x, y, color=color_list, linewidth=3)
    fig.savefig(f"pic/{filename}.png")

    if if_movie:
        makedir("video")
        def update_line(i):
            if title is not None:
                ax.set_title(f"{title[i]}")
            line.set_xdata(x[:i])
            line.set_ydata(y[:i])
            return line,
        anim = animation.FuncAnimation(fig, update_line, np.arange(0, len(x)), interval=1)
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')


def dot_line(x, y, title=None, ylim=[0, 1], xlabel="x", ylabel="y", if_movie=False, filename="dot_line"):

    makedir("pic")
    fig, ax = plt.subplots(figsize=(8, 6))
    color_list = plt.cm.tab10(4)
    ax.set_ylim(ylim)
    ax.set_title("")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    line, = ax.plot(x[-1], y[-1], "-o", color=color_list, linewidth=3)
    plt.savefig(f"pic/{filename}.png")

    if if_movie:
        makedir("video")
        def update_line(i):
            if title is not None:
                ax.set_title(f"{title[i]}")
            line.set_xdata(x[i])
            line.set_ydata(y[i])
            return line,
        anim = animation.FuncAnimation(fig, update_line, np.arange(0, len(x)), interval=1)
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')


def line_multi(x, y, title=None, xlabel="x", ylabel="y", labels=None, if_movie=False, filename="lines"):

    makedir("pic")
    fig, ax = plt.subplots(figsize=(10, 8))
    color_list = plt.cm.tab10(range(len(x)))
    ax.set_title("")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    for i in range(len(y)):
        line, = ax.plot(x[i], y[i], "-o", color=color_list[i], linewidth=3)
    if labels is not None:
        for i in range(len(labels)):
            exec('l = ax.quiver(None, None, 1, 1, color=color_list[{}], label=labels[{}])'.format(i, i))
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderpad=1)
    plt.savefig(f"pic/{filename}.png")

    if if_movie:
        makedir("video")
        def update_line(i):
            if title is not None:
                ax.set_title(f"{title[i]}")
            line.set_xdata(x[i])
            line.set_ydata(y[i])
            return line,
        anim = animation.FuncAnimation(fig, update_line, np.arange(0, len(x)), interval=1)
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'video/{filename}.mp4', writer=FFwriter)
        # anim.save(f"video/{filename}.gif", writer='pillow')
