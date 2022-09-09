import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random

np.set_printoptions(precision=8)


def vicsek_kdtree(size_bdy, num, radi, vel, eta, deltat, tmax, save_pos=False):
    """
    USAGE:
    size_bdy: square motion boundary, float, e.g. : 1.0.
    num: Particle amount for each type, int list, e.g. : [25, 75].
    vel: velocities, float list, e.g. : [0.1, 0.1].
    size_bdy: square motion boundary, float, e.g. : 1.0.
    radius: vision radii, float list, e.g. : [0.1, 0.1].
    """

    num_sum = sum(num)
    num_types = len(num)
    type_label = np.repeat(np.arange(len(num)), num)

    time_step = round(tmax / deltat)
    theta_matrix = np.zeros(shape=(num_sum, time_step))
    xpos_matrix = np.zeros(shape=(num_sum, time_step))
    ypos_matrix = np.zeros(shape=(num_sum, time_step))

    pos_all = np.random.uniform(0, size_bdy, size=(num_sum, 2))
    angle_all = np.random.uniform(-np.pi, np.pi, size=num_sum)

    theta_matrix[:, 0] = angle_all[:]
    xpos_matrix[:, 0] = pos_all[:, 0]
    ypos_matrix[:, 0] = pos_all[:, 1]

    for t in range(1, time_step):

        tree = cKDTree(pos_all, boxsize=[size_bdy, size_bdy])

        for i in range(0, num_types):
            type_idx = np.array(np.where(type_label == i)).squeeze()
            dist_mat = tree.sparse_distance_matrix(tree, max_distance=radi[i], output_type='coo_matrix')

            angle_im = np.exp(angle_all[dist_mat.col] * 1j)
            neigh = sparse.coo_matrix((angle_im, (dist_mat.row, dist_mat.col)), shape=dist_mat.get_shape())
            angle_neighed = np.squeeze(np.asarray(neigh.tocsr().mean(axis=1)))

            angle_all[type_idx] = np.angle(angle_neighed[type_idx]) + eta[i] * np.random.uniform(-np.pi, np.pi, size=num[i])

            cos, sin = np.cos(angle_all[type_idx]), np.sin(angle_all[type_idx])

            pos_all[type_idx, 0] += cos * vel[i]
            pos_all[type_idx, 1] += sin * vel[i]

        pos_all[pos_all > size_bdy] -= size_bdy
        pos_all[pos_all < 0] += size_bdy

        theta_matrix[:, t] = angle_all[:]
        xpos_matrix[:, t] = pos_all[:, 0]
        ypos_matrix[:, t] = pos_all[:, 1]

        print(t + 1, "/", time_step, end="\r")

    # if not if_anime:
    #     pass
    # else:
    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     plt.ylim(0, size_bdy)
    #     plt.xlim(0, size_bdy)
    #
    #     # qv = ax.scatter(anim_xy_pos[0, :], anim_xy_pos[1, :], marker='o', c=type_label, cmap='tab10')
    #     # qv = ax.quiver(anim_xy_pos[0, :], anim_xy_pos[1, :], np.cos(anim_angle), np.sin(anim_angle), type_label, cmap='tab10')
    #
    #     # qv = ax.scatter(xpos_matrix[:, 0], ypos_matrix[:, 0], marker='o', c=type_label, cmap='tab10')
    #     qv = ax.quiver(xpos_matrix[:, 0], ypos_matrix[:, 0], np.cos(theta_matrix[:, 0]), np.sin(theta_matrix[:, 0]), type_label, cmap='tab10')
    #
    #     def anime_generator(i):
    #         # anim_xy_pos = np.array([xpos_matrix[:, i], ypos_matrix[:, i]])
    #         # anim_angle = theta_matrix[:, i]
    #         func_input = np.c_[xpos_matrix[:, i], ypos_matrix[:, i]]
    #         qv.set_offsets(func_input)
    #         qv.set_UVC(np.cos(theta_matrix[:, i]), np.sin(theta_matrix[:, i]), type_label)
    #
    #         return qv,
    #
    #     anim = FuncAnimation(fig, anime_generator, frames=100, interval=10, blit=True, repeat=False)
    #     anim.save("test.mp4", writer='pillow')
    #     # f = r"D:/Project/20220706_VicsekModel/python/test.mp4"
    #     # writer_video = FFMpegWriter(fps=60)
    #     # anim.save(f, writer=writer_video)
    #     plt.show()

    if save_pos:
        return [theta_matrix, type_label, xpos_matrix, ypos_matrix]
    else:
        return [theta_matrix, type_label]
