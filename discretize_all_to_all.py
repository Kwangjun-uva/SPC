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

n_neuron = 425
nlayers = 2
neurons_per_layer = [25, 400]

# external current matrix
# inject external current to layer 1
stim_dim = np.sqrt(neurons_per_layer[0]).astype(int)
Iext = np.zeros((stim_dim, stim_dim))
Iext[1:4, 1:4] = 1200 * 10 ** -12
Iext[2,2] = 0

fig = plt.figure()
plt.imshow(Iext)
plt.title('Stimulus')
plt.show()

Iext_mat = np.zeros(n_neuron)
Iext_mat[:neurons_per_layer[0]] = Iext.flatten()

# connection matrix : all-to-all
conn_mat = np.ones((n_neuron, n_neuron))
conn_mat[:neurons_per_layer[0], :neurons_per_layer[0]] = 0
conn_mat[neurons_per_layer[0]:, neurons_per_layer[0]:] = 0

fig2 = plt.figure()
plt.imshow(conn_mat)
plt.title('Connection matrix')
plt.show()

# weight matrix
weights = np.random.randint(0, 20, (neurons_per_layer[1], neurons_per_layer[0]))
w_mat = np.zeros(np.shape(conn_mat))
w_mat[neurons_per_layer[0]:, :neurons_per_layer[0]] = weights
w_mat[:neurons_per_layer[0], neurons_per_layer[0]:] = weights.T

fig3 = plt.figure()
plt.imshow(w_mat)
plt.title('Weight matrix')
plt.show()

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

    # Isyn_t[:, t_i +1] = np.matmul(xtrace_t[:, t_i], conn_mat) * w_mat + Iext_mat
    Isyn_t[:, t_i + 1] = np.matmul(xtrace_t[:, t_i], w_mat) + Iext_mat
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

fig4, axs = plt.subplots(ncols=2, nrows=1,
                        figsize=(2*4, 4))

axs[0].imshow(np.mean(Isyn_t[:neurons_per_layer[0], -200:], axis=1).reshape(stim_dim, stim_dim) / (10 ** -12))
axs[1].imshow(np.mean(Isyn_t[neurons_per_layer[0]:, -200:], axis=1).reshape(20, 20) / (10 ** -12))

plt.show()