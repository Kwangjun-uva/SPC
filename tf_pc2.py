import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

AdEx = {
    't_ref': 2 * 10 ** (-3),  # ms
    'Cm': 281 * 10 ** (-12),  # pF
    'gL': 30 * 10 ** (-9),  # nS=
    'EL': -70.6 * 10 ** (-3),  # mV
    'VT': -50.4 * 10 ** (-3),  # mV
    'DeltaT': 2 * 10 ** (-3),  # mV

    # Pick an electrophysiological behaviour
    # Regular spiking (as in the paper)
    'tauw': 144 * 10 ** (-3),  # ms
    'a': 4 * 10 ** (-9),  # nS
    'b': 0.0805 * 10 ** (-9),  # nA

    # spike trace
    'x_reset': 1.,
    'I_reset': -1 * 10 ** (-12),  # pamp
    'tau_rise': 5 * 10 ** (-3),  # ms
    'tau_s': 50 * 10 ** (-3)  # ms
}

sim_dur = 1000 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-3)  # ms
num_neurons = 100
# Iext = 1200 * 10 ** -12  # pamp
Iext = tf.random.normal(shape=(num_neurons,),
                        mean=1500 * 10 ** -12, stddev=500 * 10 ** -12,
                        dtype=tf.float32,
                        name='external current')
time_steps = int(sim_dur / dt) + 1


#
# code_start = time.time()
#
# # internal variables
# v = tf.Variable(tf.ones(shape=num_neurons, dtype=tf.float32) * AdEx['EL'], name='membrane potential')
# c = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.float32), name='adaptation variable')
# ref = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.int32))
# # pre-synaptic variables
# x = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.float32), name='spike variable')
# x_tr = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.float32), name='pre-synaptic spike trace')
# # post-synaptic variable
# Isyn = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.float32), name='post-synaptic current')
# fired = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.bool))
#
#
# def dvdt(vt, ct, isynt):
#     return (dt / AdEx['Cm']) * (AdEx['gL'] * (AdEx['EL'] - vt) + AdEx['gL'] * AdEx['DeltaT'] * tf.exp(
#         (vt - AdEx['VT']) / AdEx['DeltaT']) + isynt - ct)
#
#
# def dcdt(vt, ct):
#     return (dt / AdEx['tauw']) * (AdEx['a'] * (vt - AdEx['EL']) - ct)
#
#
# def dxdt(xt):
#     return dt * (-xt / AdEx['tau_rise'])
#
#
# def dxtrdt(xt, xtrt):
#     return dt * (-xt / AdEx['tau_rise'] - xtrt / AdEx['tau_s'])
#
#
# tf_dvdt = tf.function(dvdt)
# tf_dcdt = tf.function(dcdt)
# tf_dxdt = tf.function(dxdt)
# tf_dxtrdt = tf.function(dxtrdt)
#
# vts = []
# x_trts = []
# frs = []
#
# for t in range(time_steps - 1):
#
#     ref_op = tf.greater(ref, 0)
#     v_ref = tf.where(ref_op, AdEx['EL'], v)
#     if t < 200:
#         Isyn.assign(tf.zeros(num_neurons, dtype=tf.float32))
#     elif 200 <= t < 800:
#         Isyn.assign(Iext)
#     elif t >= 800:
#         Isyn.assign(tf.zeros(num_neurons, dtype=tf.float32))
#
#     # compute changes
#     dv = tf.where(ref_op, 0.0, tf_dvdt(v_ref, c, Isyn))
#     dc = tf.where(ref_op, AdEx['b'], tf_dcdt(v_ref, c))
#     dx = tf.where(ref_op, -AdEx['x_reset'], tf_dxdt(x))
#     dxtr = tf_dxtrdt(x, x_tr)
#
#     # subtract one time step (1) from refractory vector
#     ref.assign(tf.maximum(tf.subtract(ref, 1), 0))
#
#     # update variables
#     c.assign(tf.add(c, dc))
#     x.assign(tf.add(x, dx))
#     x_tr.assign(tf.add(x_tr, dxtr))
#     v.assign(tf.add(v_ref, dv))
#
#     # update spike monitor (fired: dtype=bool): if fired = True, else = False
#     fired.assign(tf.greater_equal(v, AdEx['VT']))
#     # update refractory vector : if fired = 2, else = 0
#     ref.assign_add(tf.where(fired, int(AdEx['t_ref'] / dt), 0))
#
#     x_trts.append(x_tr.numpy())
#     frs.append(fired.numpy())
#
# x_trts = tf.stack(x_trts)
# frs = tf.stack(frs)
#
# code_end = time.time()
#
# print(code_end - code_start)


#  <<<<<< initialize with (n x t) tensor >>>>>>>
#
# # internal variables
# v = tf.Variable(tf.ones(shape=(num_neurons, time_steps), dtype=tf.float32) * AdEx['EL'], name='membrane potential')
# c = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.float32), name='adaptation variable')
# ref = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.int32))
# # pre-synaptic variables
# x = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.float32), name='spike variable')
# x_tr = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.float32), name='pre-synaptic spike trace')
# # post-synaptic variable
# Isyn = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.float32), name='post-synaptic current')
# fired = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.bool))
#
# for t in range(time_steps - 1):
#
#     ref_op = tf.greater(ref, 0)
#     v_ref = tf.where(ref_op, AdEx['EL'], v[:, t])
#     if t < 200:
#         Isyn[:, t].assign(tf.zeros(num_neurons, dtype=tf.float32))
#     elif 200 <= t < 800:
#         Isyn[:, t].assign(Iext)
#     elif t >= 800:
#         Isyn[:, t].assign(tf.zeros(num_neurons, dtype=tf.float32))
#
#     # compute changes
#     dv = tf.where(ref_op, 0.0, tf_dvdt(v_ref, c[:, t], Isyn[:, t]))
#     dc = tf.where(ref_op, AdEx['b'], tf_dcdt(v_ref, c[:, t]))
#     dx = tf.where(ref_op, -AdEx['x_reset'], tf_dxdt(x[:, t]))
#     dxtr = tf_dxtrdt(x[:, t], x_tr[:, t])
#
#     # subtract one time step (1) from refractory vector
#     ref.assign(tf.maximum(tf.subtract(ref, 1), 0))
#
#     # update variables
#     c[:, t + 1].assign(tf.add(c[:, t], dc))
#     x[:, t + 1].assign(tf.add(x[:, t], dx))
#     x_tr[:, t + 1].assign(tf.add(x_tr[:, t], dxtr))
#     v[:, t + 1].assign(tf.add(v_ref, dv))
#
#     # update spike monitor (fired: dtype=bool): if fired = True, else = False
#     fired[:, t + 1].assign(tf.greater_equal(v[:, t + 1], AdEx['VT']))
#     # update refractory vector : if fired = 2, else = 0
#     ref.assign_add(tf.where(fired[:, t + 1], int(AdEx['t_ref'] / dt), 0))

# A basic LIF neuron
class AdEx_Layer(object):

    def __init__(self, neuron_model_constants, n_neuron, dt):
        for key in neuron_model_constants:
            setattr(self, key, neuron_model_constants[key])

        self.num_neurons = n_neuron
        self.dt = dt

        # internal variables
        self.v = tf.Variable(tf.ones(self.num_neurons, dtype=tf.float32) * self.EL)
        self.c = tf.Variable(tf.zeros(self.num_neurons, dtype=tf.float32))
        self.ref = tf.Variable(tf.zeros(self.num_neurons, dtype=tf.int32))
        # pre-synaptic variables
        self.x = tf.Variable(tf.zeros(self.num_neurons, dtype=tf.float32))
        self.x_tr = tf.Variable(tf.zeros(self.num_neurons, dtype=tf.float32))
        # post-synaptic variable
        self.Isyn = tf.Variable(tf.zeros(self.num_neurons, dtype=tf.float32))
        self.fired = tf.Variable(tf.zeros(self.num_neurons, dtype=tf.bool))

        # # over time
        # self.spikes = []

    def __call__(self, I_ext):  # sim_duration, I_ext, w_matrix):

        # self.sim_dur = sim_duration
        # self.timestep = int(sim_duration / self.dt)

        # self.w = w_matrix

        self.update_var(I_ext)

        return self.fired

    def update_var(self, I_ext):
        # for t in range(self.timestep - 1):
        self.Isyn = I_ext

        ref_constraint = tf.greater(self.ref, 0)
        self.v = tf.where(ref_constraint, self.EL, self.v)

        # if t < 200:
        #     self.Isyn = tf.zeros(self.num_neurons, dtype=tf.float32)
        # elif 200 <= t < 800:
        #     self.Isyn = I_ext
        # elif t >= 800:
        #     self.Isyn = tf.zeros(self.num_neurons, dtype=tf.float32)

        # # compute changes
        # dv = tf.where(self.ref, 0.0, dvdt(self.v))
        # dc = tf.where(self.ref, self.b], dcdt(self.v, self.c))
        # dx = tf.where(self.ref, - self.x_reset], dxdt(self.x))
        # dxtr = tf_dxtrdt(self.x, self.x_tr)
        #
        # update variables
        # self.c.assign(tf.add(self.c, dc))
        # self.x.assign(tf.add(self.x, dx))
        # self.x_tr.assign(tf.add(self.x_tr, self.dxtr))
        # self.v.assign(tf.add(self.v, dv))
        self.update_v(ref_constraint)
        self.update_c(ref_constraint)
        self.update_x(ref_constraint)
        self.update_xtr()

        # subtract one time step (1) from refractory vector
        # self.ref.assign(tf.maximum(tf.subtract(self.ref, 1), 0))
        self.ref = tf.maximum(tf.subtract(self.ref, 1), 0)

        # update spike monitor (fired: dtype=bool): if fired = True, else = False
        # self.fired.assign(tf.greater_equal(self.v, self.VT))
        self.fired = tf.greater_equal(self.v, self.VT)
        # update refractory vector : if fired = 2, else = 0
        self.ref.assign_add(tf.where(self.fired, int(self.t_ref / self.dt), 0))

        # self.spikes.append(self.fired.numpy())

        # self.spikes = tf.stack(self.spikes)

    def update_v(self, ref_constraint):
        dv = (self.dt / self.Cm) * (self.gL * (self.EL - self.v) +
                                    self.gL * self.DeltaT * tf.exp((self.v - self.VT) / self.DeltaT) +
                                    self.Isyn - self.c)
        # dv_func = tf.function(dv)
        # dv_ref = tf.where(self.ref, 0.0, dv_func(self.v))
        dv_ref = tf.where(ref_constraint, 0.0, dv)
        return tf.add(self.v, dv_ref)

    def update_c(self, ref_constraint):
        dc = (self.dt / self.tauw) * (self.a * (self.v - self.EL) - self.c)
        # dc_func = tf.function(dc)
        # dc_ref = tf.where(self.ref, 0.0, dc_func(self.c))
        dc_ref = tf.where(ref_constraint, 0.0, dc)
        return tf.add(self.c, dc_ref)

    def update_x(self, ref_constraint):
        dx = self.dt * (-self.x / self.tau_rise)
        # dx_func = tf.function(dx)
        # dx_ref = tf.where(self.ref, 0.0, dx_func(self.x))
        dx_ref = tf.where(ref_constraint, 0.0, dx)
        return tf.add(self.x, dx_ref)

    def update_xtr(self):
        dxtr = self.dt * (-self.x / self.tau_rise - self.x_tr / self.tau_s)
        # dxtr_func = tf.function(dxtr)
        # return tf.add(self.xtr, dxtr_func(dxtr))
        return tf.add(self.x_tr, dxtr)


adex_01 = AdEx_Layer(neuron_model_constants=AdEx,
                     n_neuron=num_neurons,
                     dt=dt)

ext_current = tf.random.normal(shape=(num_neurons,),
                               mean=1500 * 10 ** -12, stddev=500 * 10 ** -12,
                               dtype=tf.float32,
                               name='external current')
adex_01(ext_current)
# wsyn = 550 * 10 ** -12
# n_neuron = 32
# nlayers = 2
# neurons_per_layer = [16, 16]
#
# # external current matrix
# Iext_mat = np.zeros(n_neuron)
# # inject external current to layer 1
# Iext = np.arange(0, 1600, 100) * 10 ** -12
# Iext_mat[:neurons_per_layer[0]] = Iext
#
# # connection matrix
# conn_mat = np.zeros((n_neuron, n_neuron))
# for i in range(n_neuron):
#     if i < neurons_per_layer[0]:
#         conn_mat[i, i+neurons_per_layer[0]] = 1

# t = np.arange(t_step + 1)
#
# c_t = np.zeros((n_neuron, t_step + 1))
# v_t = np.zeros((n_neuron, t_step + 1))
#
# x_t = np.zeros((n_neuron, t_step + 1))
# xtrace_t = np.zeros((n_neuron, t_step + 1))
#
# Isyn_t = np.zeros((n_neuron, t_step +1))
# spike_monitor = np.zeros((n_neuron, t_step + 1))
#
# v_t[:, 0] = EL
#
# ref = np.zeros(n_neuron)
#
# for t_i in range(t_step):
#
#     Isyn_t[:, t_i +1] = np.matmul(xtrace_t[:, t_i], conn_mat) * wsyn + Iext_mat
#     v_t[:, t_i + 1] = v_t[:, t_i] + dvdt(v_t[:, t_i], c_t[:, t_i], Isyn_t[:, t_i + 1]) * [ri == 0 for ri in ref]
#     c_t[:, t_i + 1] = c_t[:, t_i] + dcdt(c_t[:, t_i], v_t[:, t_i]) * [ri == 0 for ri in ref]
#
#     ref = np.asarray([np.max([ref[i]-1, 0]) for i in range(len(ref))])
#
#     x_t[:, t_i + 1] = x_t[:, t_i] + dxdt(x_t[:, t_i])
#     xtrace_t[:, t_i + 1] = xtrace_t[:, t_i] + dxtrdt(xtrace_t[:, t_i], x_t[:, t_i])
#
#     # reset upon reaching the threshold membrane potential
#     spiking_bool = v_t[:, t_i + 1] >= VT
#     v_t[spiking_bool, t_i + 1] = EL
#     c_t[spiking_bool, t_i + 1] += b
#     x_t[spiking_bool, t_i + 1] = -x_reset
#
#     spike_monitor[:, t_i] = spiking_bool.astype(int)
#     ref += (t_ref / dt) * spiking_bool
#
# fig, axs = plt.subplots(ncols=2, nrows=1,
#                         figsize=(2*4, 4))
#
# axs[0].scatter(Iext / (10 ** -12), np.array([np.mean(xtrace_t[i, -200:]) for i in range(neurons_per_layer[0])]) * wsyn / (10 ** -12))
# axs[1].scatter(Iext / (10 ** -12), np.array([np.mean(Isyn_t[i, -200:]) for i in range(neurons_per_layer[0], n_neuron)]) / (10 ** -12))
#
# plt.show()
