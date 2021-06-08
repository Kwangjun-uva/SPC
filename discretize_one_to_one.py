import numpy as np
import matplotlib.pyplot as plt


t_ref = 2 * 10 ** (-3) # ms
Cm = 281 * 10 ** (-12) # pF
gL = 30 * 10 ** (-9) # nS=
EL = -70.6 * 10 ** (-3) # mV
VT = -50.4 * 10 ** (-3) # mV
DeltaT = 2 * 10 ** (-3) # mV

# Pick an electrophysiological behaviour
## Regular spiking (as in the paper)
tauw = 144 * 10 ** (-3) # ms
a = 4 * 10 ** (-9) # nS
b = 0.0805 * 10 ** (-9) # nA

# spike trace
x_reset = 1.
I_reset = -1 * 10 ** (-12)  # pamp
tau_rise = 5 * 10 ** (-3) # ms
tau_s = 50 * 10 ** (-3) # ms

# neuron eqs
def dvdt(v, c, isyn):
    return (dt / Cm) * (gL * (EL - v) + gL * DeltaT * np.exp((v - VT) / DeltaT) + isyn - c)
def dcdt(c, v):
    return (dt / tauw) * (a * (v - EL) - c)

# syn eqs
def dxdt(x):
    return dt * (-x / tau_rise)
def dxtrdt(xtr, x):
    return dt * (-x / tau_rise - xtr / tau_s)

sim_dur = 1000 * 10 ** (-3) # ms
dt = 1 * 10 ** (-3) # ms
t_step = int(sim_dur / dt)

wsyn = 550 * 10 ** -12
n_neuron = 32
nlayers = 2
neurons_per_layer = [16, 16]

# external current matrix
Iext_mat = np.zeros(n_neuron)
# inject external current to layer 1
Iext = np.arange(0, 1600, 100) * 10 ** -12
Iext_mat[:neurons_per_layer[0]] = Iext

# connection matrix
conn_mat = np.zeros((n_neuron, n_neuron))
for i in range(n_neuron):
    if i < neurons_per_layer[0]:
        conn_mat[i, i+neurons_per_layer[0]] = 1

# for i in range(n_neuron):
#     if i < neurons_per_layer[0]:
#         conn_mat[i, neurons_per_layer[0]] = 1

t = np.arange(t_step + 1)

c_t = np.zeros((n_neuron, t_step + 1))
v_t = np.zeros((n_neuron, t_step + 1))

x_t = np.zeros((n_neuron, t_step + 1))
xtrace_t = np.zeros((n_neuron, t_step + 1))

Isyn_t = np.zeros((n_neuron, t_step +1))
spike_monitor = np.zeros((n_neuron, t_step + 1))

v_t[:, 0] = EL

ref = np.zeros(n_neuron)

for t_i in range(t_step):

    Isyn_t[:, t_i +1] = np.matmul(xtrace_t[:, t_i], conn_mat) * wsyn + Iext_mat
    v_t[:, t_i + 1] = v_t[:, t_i] + dvdt(v_t[:, t_i], c_t[:, t_i], Isyn_t[:, t_i + 1]) * [ri == 0 for ri in ref]
    c_t[:, t_i + 1] = c_t[:, t_i] + dcdt(c_t[:, t_i], v_t[:, t_i]) * [ri == 0 for ri in ref]

    ref = np.asarray([np.max([ref[i]-1, 0]) for i in range(len(ref))])

    x_t[:, t_i + 1] = x_t[:, t_i] + dxdt(x_t[:, t_i])
    xtrace_t[:, t_i + 1] = xtrace_t[:, t_i] + dxtrdt(xtrace_t[:, t_i], x_t[:, t_i])

    # reset upon reaching the threshold membrane potential
    spiking_bool = v_t[:, t_i + 1] >= VT
    v_t[spiking_bool, t_i + 1] = EL
    c_t[spiking_bool, t_i + 1] += b
    x_t[spiking_bool, t_i + 1] = -x_reset

    spike_monitor[:, t_i] = spiking_bool.astype(int)
    ref += (t_ref / dt) * spiking_bool

fig, axs = plt.subplots(ncols=2, nrows=1,
                        figsize=(2*4, 4))

axs[0].scatter(Iext / (10 ** -12), np.array([np.mean(xtrace_t[i, -200:]) for i in range(neurons_per_layer[0])]) * wsyn / (10 ** -12))
axs[1].scatter(Iext / (10 ** -12), np.array([np.mean(Isyn_t[i, -200:]) for i in range(neurons_per_layer[0], n_neuron)]) / (10 ** -12))

plt.show()

# def neuron_eqs(n_neurons, conn_matx, dur_step, Iext, w_syn):
#
#     t = np.arange(dur_step + 1)
#
#     c_t = np.zeros((n_neurons, dur_step + 1))
#     v_t = np.zeros((n_neurons, dur_step + 1))
#
#     x_t = np.zeros((n_neurons, dur_step + 1))
#     xtrace_t = np.zeros((n_neurons, dur_step + 1))
#
#     Isyn_t = np.zeros((n_neurons, dur_step +1))
#     spike_monitor = np.zeros((n_neurons, dur_step + 1))
#
#     v_t[:, 0] = EL
#
#     ref = np.zeros(n_neurons)
#
#     for t_i in range(dur_step):
#
#         Isyn_t[:, t_i +1] = np.matmul(xtrace_t[:, t_i], conn_matx) * w_syn + Iext * ext_curr_idx
#         v_t[:, t_i + 1] = v_t[:, t_i] + dvdt(v_t[:, t_i], c_t[:, t_i], Isyn_t[:, t_i + 1]) * [ri == 0 for ri in ref]
#         c_t[:, t_i + 1] = c_t[:, t_i] + dcdt(c_t[:, t_i], v_t[:, t_i]) * [ri == 0 for ri in ref]
#
#         ref = np.asarray([np.max([ref[i]-1, 0]) for i in range(len(ref))])
#
#         x_t[:, t_i + 1] = x_t[:, t_i] + dxdt(x_t[:, t_i])
#         xtrace_t[:, t_i + 1] = xtrace_t[:, t_i] + dxtrdt(xtrace_t[:, t_i], x_t[:, t_i])
#
#         # reset upon reaching the threshold membrane potential
#         spiking_bool = v_t[:, t_i + 1] >= VT
#         v_t[spiking_bool, t_i + 1] = EL
#         c_t[spiking_bool, t_i + 1] += b
#         x_t[spiking_bool, t_i + 1] = -x_reset
#
#         spike_monitor[:, t_i] = spiking_bool.astype(int)
#         ref += (t_ref / dt) * spiking_bool
#
#     return t, v_t, xtrace_t, spike_monitor, Isyn_t

# # Iexts = np.zeros(n_neuron)
# # Iexts[:10] = np.arange(0, 1000, 100) * 10**-12
# tt, vv, xx, spm, Isyn = neuron_eqs(n_neurons=n_neuron,
#                                    conn_matx=conn_mat,
#                                    dur_step=t_step,
#                                    Iext=np.array([1350, 1250, 1600, 0]) * 10**-12,
#                                    w_syn=wsyn)
# #
# # plot data
# fig, axs = plt.subplots(nrows=n_neuron, ncols=3, figsize=(3*4, n_neuron*4))
# for i in range(n_neuron):
#     axs[i, 0].plot(tt, Isyn[i] / (10 ** -12), c='b', label='mean = {0:.2f}'.format(np.mean(Isyn[i][-200:]) / (10 ** -12)))
#     axs[i, 0].set_title('Isyn(t) of neuron #{0:.2f}'.format(i+1))
#     axs[i, 0].legend()
#     axs[i, 1].plot(tt, vv[i] / (10 ** -3), c='b')
#     axs[i, 1].set_title('v(t) of neuron #{0:.2f}'.format(i+1))
#     axs[i, 2].plot(tt, xx[i] * wsyn / (10 ** -12), c='g', label='mean = {0:.2f}\nnSpike = {1}'.format(np.mean(xx[i][-200:]) * wsyn / (10 ** -12), np.sum(spm[i])))
#     axs[i, 2].scatter([j for j,n in enumerate(spm[i]) if n > 0], np.zeros(int(spm[i].sum())), c='r', marker='o')
#     axs[i, 2].legend()
#     axs[i, 2].set_title('x(t) of neuron #{0:.2f}'.format(i+1))
# # plt.tight_layout()
# plt.show()