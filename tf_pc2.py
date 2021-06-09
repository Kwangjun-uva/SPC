import numpy as np
import tensorflow as tf

# import time

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
# Iext = tf.random.normal(shape=(num_neurons,),
#                         mean=1500 * 10 ** -12, stddev=500 * 10 ** -12,
#                         dtype=tf.float64,
#                         name='external current')
time_steps = int(sim_dur / dt) + 1


#
# code_start = time.time()
#
# # internal variables
# v = tf.Variable(tf.ones(shape=num_neurons, dtype=tf.float64) * AdEx['EL'], name='membrane potential')
# c = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.float64), name='adaptation variable')
# ref = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.int32))
# # pre-synaptic variables
# x = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.float64), name='spike variable')
# x_tr = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.float64), name='pre-synaptic spike trace')
# # post-synaptic variable
# Isyn = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.float64), name='post-synaptic current')
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
#         Isyn.assign(tf.zeros(num_neurons, dtype=tf.float64))
#     elif 200 <= t < 800:
#         Isyn.assign(Iext)
#     elif t >= 800:
#         Isyn.assign(tf.zeros(num_neurons, dtype=tf.float64))
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
# v = tf.Variable(tf.ones(shape=(num_neurons, time_steps), dtype=tf.float64) * AdEx['EL'], name='membrane potential')
# c = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.float64), name='adaptation variable')
# ref = tf.Variable(tf.zeros(shape=num_neurons, dtype=tf.int32))
# # pre-synaptic variables
# x = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.float64), name='spike variable')
# x_tr = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.float64), name='pre-synaptic spike trace')
# # post-synaptic variable
# Isyn = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.float64), name='post-synaptic current')
# fired = tf.Variable(tf.zeros(shape=(num_neurons, time_steps), dtype=tf.bool))
#
# for t in range(time_steps - 1):
#
#     ref_op = tf.greater(ref, 0)
#     v_ref = tf.where(ref_op, AdEx['EL'], v[:, t])
#     if t < 200:
#         Isyn[:, t].assign(tf.zeros(num_neurons, dtype=tf.float64))
#     elif 200 <= t < 800:
#         Isyn[:, t].assign(Iext)
#     elif t >= 800:
#         Isyn[:, t].assign(tf.zeros(num_neurons, dtype=tf.float64))
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

    def __init__(self, neuron_model_constants, n_layers, n_neuron, dt, Iext):
        """

        :param neuron_model_constants: dict
        :param n_layers: int
        :param n_neuron: list
        :param dt: float
        :param Iext: float, list, or array
        """

        for key in neuron_model_constants:
            setattr(self, key, neuron_model_constants[key])

        # network architecture
        self.num_layers = n_layers
        self.num_neurons = n_neuron

        # simulation parameters
        self.dt = dt
        self._step = 0

        # internal variables
        self.n_variable = sum(self.num_neurons)
        self.v = tf.Variable(tf.ones(self.n_variable, dtype=tf.float64) * self.EL)
        self.c = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))
        self.ref = tf.Variable(tf.zeros(self.n_variable, dtype=tf.int32))
        # pre-synaptic variables
        self.x = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))
        self.x_tr = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))
        # post-synaptic variable
        self.Isyn = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))
        self.fired = tf.Variable(tf.zeros(self.n_variable, dtype=tf.bool))

        self.conn_mat = np.zeros((self.n_variable, self.n_variable))

        # constant weight
        self.w_const = 550 * 10 ** -12

        # feed external corrent to the first layer
        Iext_np = np.zeros(self.n_variable)
        Iext_np[:self.num_neurons[0]] = Iext
        self.Iext = tf.convert_to_tensor(Iext_np)

    def __call__(self, I_ext):

        self.update_var()

        self._step += 1

        return self.fired.numpy()

    def update_var(self):

        if self._step == 799:

            self.Isyn_pre = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))
            self.Isyn_post = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))

        elif self._step >= 800:
            self.Isyn_pre += self.Isyn

        # feed synaptic current to higher layers
        self.update_Isyn()

        ref_constraint = tf.greater(self.ref, 0)
        self.v = tf.where(ref_constraint, self.EL, self.v)

        self.v = self.update_v(ref_constraint)
        self.c = self.update_c(ref_constraint)
        self.x = self.update_x(ref_constraint)
        self.x_tr = self.update_xtr()

        # subtract one time step (1) from refractory vector
        self.ref = tf.maximum(tf.subtract(self.ref, 1), 0)

        # update spike monitor (fired: dtype=bool): if fired = True, else = False
        self.fired = tf.greater_equal(self.v, self.VT)
        # update refractory vector : if fired = 2, else = 0
        self.ref = tf.add(self.ref, tf.where(self.fired, int(self.t_ref / self.dt), 0))

    def update_v(self, ref_constraint):
        dv = (self.dt / self.Cm) * (self.gL * (self.EL - self.v) +
                                    self.gL * self.DeltaT * tf.exp((self.v - self.VT) / self.DeltaT) +
                                    self.Isyn - self.c)

        dv_ref = tf.where(ref_constraint, 0.0, dv)
        return tf.add(self.v, dv_ref)

    def update_c(self, ref_constraint):
        dc = (self.dt / self.tauw) * (self.a * (self.v - self.EL) - self.c)
        dc_ref = tf.where(ref_constraint, 0.0, dc)
        return tf.add(self.c, dc_ref)

    def update_x(self, ref_constraint):
        dx = self.dt * (-self.x / self.tau_rise)
        dx_ref = tf.where(ref_constraint, 0.0, dx)
        return tf.add(self.x, dx_ref)

    def update_xtr(self):
        dxtr = self.dt * (-self.x / self.tau_rise - self.x_tr / self.tau_s) * self.w_const
        return tf.add(self.x_tr, dxtr)

    def update_Isyn(self):
        # return tf.add(tf.einsum('nm,m->n', self.w, self.x_tr), self.Iext)
        return tf.add(self.w @ self.x_tr, self.Iext)

    def get_current_timestep(self):
        return self._step * self.dt

    def connect_reset(self):
        self.conn_mat = np.zeros((self.n_variable, self.n_variable))

    def connect_by_layer(self, source, target, conn_type='FC'):
        source_idx = sum(self.num_neurons[:source - 1])
        target_idx = sum(self.num_neurons[:target - 1])
        if conn_type == 'FC':
            self.conn_mat[source_idx: source_idx + self.num_neurons[source - 1],
                target_idx: target_idx + self.num_neurons[target - 1]] = 1
        elif conn_type == 'one-to-one':
            self.conn_mat[source_idx: source_idx + self.num_neurons[source - 1],
                target_idx: target_idx + self.num_neurons[target - 1]] = np.identity(self.num_neurons[source - 1])

    def connect_fc_all(self):
        connect_fully = [self.connect(i, j) for i in range(1, 5) for j in range(1, 5) if i != j]

    def initialize_weight(self):
        np_weights = np.zeros(self.conn_mat.shape)
        for i in range(self.num_layers - 1):
            pre_begin, pre_end = (sum(self.num_neurons[:i]), sum(self.num_neurons[:i + 1]))
            post_begin, post_end = (sum(self.num_neurons[:i + 1]), sum(self.num_neurons[:i + 2]))

            np_weights[pre_begin:pre_end, post_begin:post_end] = np.random.normal(1.0, 1.0, (
                self.num_neurons[i], self.num_neurons[i + 1])) * self.conn_mat[pre_begin:pre_end, post_begin:post_end]
            np_weights[post_begin:post_end, pre_begin:pre_end] = np_weights[pre_begin:pre_end, post_begin:post_end].T

        self.w = tf.convert_to_tensor(np_weights)

        # self.w = tf.Variable(tf.where(tf.convert_to_tensor(self.conn_mat.astype(bool)),
        #                               tf.random.normal(self.conn_mat.shape, 1.0, 1.0),
        #                               0.0))

    # def weight_update(self):
    #     self.Isyn_pre = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))
    #     self.Isyn_post = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))


adex_01 = AdEx_Layer(neuron_model_constants=AdEx,
                     n_layers=15,
                     n_neuron=[3, 3],
                     dt=dt,
                     Iext=np.random.normal(loc=1500 * 10 ** -12, scale=500 * 10 ** -12, size=3))

# ext_current = tf.random.normal(shape=(num_neurons,),
#                                mean=1500 * 10 ** -12, stddev=500 * 10 ** -12,
#                                dtype=tf.float64,
#                                name='external current')
# ext_current = tf.range(0, 1600, 1600/100, dtype=tf.float64) * 10 ** -12

# frs = []
# for ti in range(time_steps):
#     frs.append(adex_01(ext_current))
#
# frs = tf.convert_to_tensor(frs)
# # print (tf.reduce_sum(tf.cast(frs, tf.int32), axis=0))
# plt.scatter(ext_current.numpy(), tf.reduce_sum(tf.cast(frs, tf.int32), axis=0).numpy())
# plt.show()


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
