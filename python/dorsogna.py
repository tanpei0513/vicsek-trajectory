import numpy as np
import matplotlib.pyplot as plt
# import scipy, persim
# from ripser import ripser
from scipy.integrate import ode

from IPython.display import set_matplotlib_formats

# set_matplotlib_formats('png', 'pdf')


def dorsogna(t, Z, alpha, beta, cA, cR, lA, lR):
    '''
    Gets derivatives of position and velocity of particles according to the
    D'Orsogna model of soft-core particle interactions.

    Inputs:
        t: (unused) time, for integrator use only
        Z: (1-d ndarray, 4*num_cells long) current step position and velocity

    Parameters:
        alpha: self-propulsion coefficient
        beta: friction coefficient
        cA: attractive amplitude
        cR: repulsive amplitude
        lA: attractive range
        lR: repulsive range

    Output: derivative of position and velocity at current step to be
    integrated to obtain position and velocity at next step
    '''
    meps = np.finfo(np.float64).eps
    # Get ICs from input vector
    num_cells = len(Z) // 4
    x = Z[0:num_cells][None, :]
    y = Z[num_cells:2 * num_cells][None, :]
    vx = Z[2 * num_cells:3 * num_cells]
    vy = Z[3 * num_cells:]

    # Compute model components
    xdiff = x - x.T
    ydiff = y - y.T
    D = np.sqrt(xdiff ** 2 + ydiff ** 2)

    with np.errstate(over='raise'):
        v_normSq = vx ** 2 + vy ** 2

    u_prime = - cA / lA * np.exp(-D / lA) + cR / lR * np.exp(-D / lR)

    dvxdt = (alpha - beta * v_normSq) * vx - np.sum(u_prime * xdiff / (D + meps), axis=1)
    dvydt = (alpha - beta * v_normSq) * vy - np.sum(u_prime * ydiff / (D + meps), axis=1)

    output = np.hstack((vx, vy, dvxdt, dvydt))

    return output


def ode_rk4(ic_vec, t0, tf, parameters=[1.5, 0.5, 0.1, 1, .1, 1]):
    # Simulate position and velocity until last desired frame using RK4/5
    simu = [ic_vec]
    r = ode(dorsogna).set_integrator('dopri5', atol=10 ** (-3))
    r.set_initial_value(ic_vec, t0).set_f_params(*parameters)
    while r.successful() and r.t < tf:
        print("Simulating frame", int(r.t + 1))
        simu.append(r.integrate(r.t + 1))
    return simu



#number of agents
num_cells = 500

#intialization
ic_vec = np.random.uniform(-1,1,4*num_cells)

#double ring
alpha,beta,cA,cR,lA,lR = 1.5,0.5,1.0,0.85,1.0,0.85
#single mill
# alpha,beta,cA,cR,lA,lR = 1.5,0.5,1.0,0.5,1.0,0.1
#double mill
#alpha,beta,cA,cR,lA,lR = 1.5,0.5,1.0,0.9,1.0,0.5
#swarm
# alpha,beta,cA,cR,lA,lR = 1.5,0.5,1.0,0.1,1.0,0.5
#Escape
# alpha,beta,cA,cR,lA,lR = 1.5,0.5,1.0,2.0,1.0,0.9

tmax = 100
parameters=[alpha,beta,cA,cR,lA,lR]
simulated_cells = ode_rk4(ic_vec,0,tmax,parameters=parameters)


# final_time = simulated_cells[5]

#extract positions and velocities
x = simulated_cells[-1][:num_cells]
y = simulated_cells[-1][num_cells:2*num_cells]
vx = simulated_cells[-1][2*num_cells:3*num_cells]
vy = simulated_cells[-1][3*num_cells:]



#determine clockwise, counter clockwise-moving agents
u = np.vstack([x - np.mean(x),y - np.mean(y)])
v = np.vstack([vx,vy])
w=np.arctan2(u[0,:]*v[1,:]-u[1,:]*v[0,:],u[0,:]*v[0,:]+u[1,:]*v[1,:])


# x = simulated_cells[-1][:num_cells]
# y = simulated_cells[-1][num_cells:2*num_cells]
# vx = simulated_cells[-1][2*num_cells:3*num_cells]
# vy = simulated_cells[-1][3*num_cells:]

# #plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# # nx = (x-min(x))/(max(x)-min(x))
# # ny = (y-min(y))/(max(y)-min(y))
#
# ax.scatter(x,y,c="k")
# ax.quiver(x[w<0],y[w<0],vx[w<0],vy[w<0],color="r",label = "Clockwise")
# ax.quiver(x[w>=0],y[w>=0],vx[w>=0],vy[w>=0],color="b",label = "Counter Clockwise")
#
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax.set_xlim([min(x) * 1.2,max(x) * 1.2])
# # ax.set_ylim([min(y) * 1.2,max(y) * 1.2])
#
# # ax.set_title("Swarming population",fontsize=30)
# # plt.legend()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams.update({'figure.max_open_warning': 0,
                     'font.size': 12,
                     'animation.ffmpeg_path': "D:/Project/20220706_VicsekModel/python_test/venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe"})




fig, ax = plt.subplots(figsize=(16, 12))
# ax.set_xlim([min(x) * 1.2,max(x) * 1.2])
# ax.set_ylim([min(y) * 1.2,max(y) * 1.2])
ax.set_title("Time at :{}".format(0))
qv_args = dict(units='width', pivot='tail', width=0.005, headaxislength=8, headlength=8, headwidth=4)
# qv = ax.quiver(simulated_cells[-1][:num_cells],
#                simulated_cells[-1][num_cells:2*num_cells],
#                simulated_cells[-1][2*num_cells:3*num_cells],
#                simulated_cells[-1][3*num_cells:],color='red')
color = plt.cm.tab10(range(2))
qv0=ax.quiver(simulated_cells[-1][:num_cells][w<0],
              simulated_cells[-1][num_cells:2*num_cells][w<0],
              simulated_cells[-1][2 * num_cells:3 * num_cells][w<0],
              simulated_cells[-1][3*num_cells:][w<0],color=color[0],label = "Clockwise", **qv_args)
qv1=ax.quiver(simulated_cells[-1][:num_cells][w>=0],simulated_cells[-1][num_cells:2*num_cells][w>=0],simulated_cells[-1][2 * num_cells:3 * num_cells][w>=0],
              simulated_cells[-1][3*num_cells:][w>=0],color=color[1],label = "Counter Clockwise", **qv_args)
# qv = ax.scatter(simulated_cells[-1][:num_cells],
#                simulated_cells[-1][num_cells:2*num_cells], color="black", s=5)
# Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderpad=1)
plt.show()


def update_plotArrow(i):
    ax.set_title("Time at: {}".format(i + 1))
    qv0.set_offsets(np.c_[simulated_cells[i][:num_cells][w<0],
               simulated_cells[i][num_cells:2*num_cells][w<0]])
    qv0.set_UVC(simulated_cells[i][2*num_cells:3*num_cells][w<0],
               simulated_cells[i][3*num_cells:][w<0])
    qv1.set_offsets(np.c_[simulated_cells[i][:num_cells][w>=0],
               simulated_cells[i][num_cells:2*num_cells][w>=0]])
    qv1.set_UVC(simulated_cells[i][2*num_cells:3*num_cells][w>=0],
               simulated_cells[i][3*num_cells:][w>=0])
    return qv0, qv1,

anim = animation.FuncAnimation(fig, update_plotArrow, np.arange(0, 100), interval=1)
FFwriter = animation.FFMpegWriter(fps=10)
anim.save('video/dorsogna2.mp4', writer=FFwriter)
# anim.save("video/particle_motion_movie.gif", writer='pillow')


from sklearn.decomposition import PCA
from plot import scatter_plot

# time_series = [*np.arange(1,200)]
time_series = [*np.arange(1,100),*np.arange(100, tmax, 100)]

pca1 = [None] * len(time_series)
pca2 = [None] * len(time_series)
accuracy_arry = [None] * len(time_series)

for i,t in enumerate(time_series):
    vx = [simulated_cells[j][2*num_cells:3*num_cells] for j in range(t)]
    vy = [simulated_cells[j][3*num_cells:] for j in range(t)]
    vx = np.array(vx); vy = np.array(vy);

    pca_data = np.vstack([vx,vy]).T
    # pca_data_sd = StandardScaler().fit_transform(pca_data)
    components = PCA(n_components=2,random_state=31415).fit_transform(pca_data)
    pca1[i] = components[:,0]
    pca2[i] = components[:,1]
    # cluster_label = KMeans(n_clusters=2).fit(components).labels_

scatter_plot(x=pca1, y=pca2, xlabel="PCA1", ylabel="PCA2",
             type_list=[w<0]*len(time_series),labels=["type1","type2"],
             if_movie=True, filename="dorsogna_pca")