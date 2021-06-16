import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import time

##List all your physical GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

# A basic adex LIF neuron
class AdEx_Layer(object):

    def __init__(self, neuron_model_constants, n_layers, n_neuron, n_pc_layers, n_pred_neurons, n_stim):
        """
        :param neuron_model_constants: dict
        :param n_layers: int
        :param n_neuron: list
        :param n_pc_layers: int
        :param n_pred_neurons: list
        """

        for key in neuron_model_constants:
            setattr(self, key, neuron_model_constants[key])

        # network architecture
        self.num_layers = n_layers
        self.num_neurons = n_neuron
        self.n_pc_layers = int((n_layers + 1) / 2)

        self.n_groups = n_pc_layers * 3 + 1
        self.neurons_per_group = [n_stim] * 3 + np.repeat([n_pred_neurons[:-1]], 3).tolist() + n_pred_neurons[-1]

        # internal variables
        # self.n_variable = sum(self.num_neurons)
        self.n_variable = sum(self.neurons_per_group)
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

        # weight update
        self.l_time = None

        self.w = None
        self.w_init = None
        self.xtr_record = None

    def __call__(self, sim_duration, time_step, lt, I_ext):

        # simulation parameters
        self.T = sim_duration
        self.dt = time_step
        self._step = 0
        self.l_time = lt

        # feed external corrent to the first layer
        self.Iext = self.create_Iext(I_ext)

        self.fs = tf.Variable(tf.zeros(sum(self.num_neurons), dtype=tf.int64))
        # fr = []

        for t in range(int(self.T/self.dt)):

            # update internal variable (v, c, x, x_tr)
            self.update_var()
            # update synaptic variable (Isyn = w * x_tr + Iext)
            self.update_Isyn()
            self.record_pre_post()

            # save firing rate (fs) and firing time (fr)
            fired_int = tf.cast(self.fired, dtype=tf.int64)
            self.fs.assign_add(fired_int)
            # fr.append(fired_int.numpy())

            self._step += 1

    def create_Iext(self, Iext):

        Iext_np = np.zeros(self.n_variable)
        Iext_np[:self.num_neurons[0]] = Iext

        return tf.constant(Iext_np)

    def update_var(self):

        # feed synaptic current to higher layers
        self.Isyn = self.update_Isyn()

        ref_constraint = tf.greater(self.ref, 0)
        self.v = tf.where(ref_constraint, self.EL, self.v)

        self.v = self.update_v(ref_constraint)
        self.c = self.update_c(ref_constraint)
        self.x = self.update_x()
        self.x_tr = self.update_xtr()

        # subtract one time step (1) from refractory vector
        self.ref = tf.maximum(tf.subtract(self.ref, 1), 0)

        # update spike monitor (fired: dtype=bool): if fired = True, else = False
        self.fired = tf.greater_equal(self.v, self.VT)
        self.x = tf.where(self.fired, -self.x_reset, self.x)
        # update refractory vector : if fired = 2, else = 0
        self.ref = tf.add(self.ref, tf.where(self.fired, int(self.t_ref / self.dt), 0))

        # save Isyn

        # return

    def update_v(self, ref_constraint):
        dv = (self.dt / self.Cm) * (self.gL * (self.EL - self.v) +
                                    self.gL * self.DeltaT * tf.exp((self.v - self.VT) / self.DeltaT) +
                                    self.Isyn - self.c)
        dv_ref = tf.where(ref_constraint, 0.0, dv)
        return tf.add(self.v, dv_ref)

    def update_c(self, ref_constraint):
        dc = (self.dt / self.tauw) * (self.a * (self.v - self.EL) - self.c)
        dc_ref = tf.where(ref_constraint, self.b, dc)
        return tf.add(self.c, dc_ref)

    def update_x(self):
        dx = self.dt * (-self.x / self.tau_rise)
        return tf.add(self.x, dx)

    def update_xtr(self):
        dxtr = self.dt * (-self.x / self.tau_rise - self.x_tr / self.tau_s) * self.w_const
        return tf.add(self.x_tr, dxtr)

    def update_Isyn(self):
        return tf.tensordot(self.w * self.conn_mat, self.x_tr, 1) + self.Iext

    def get_current_timestep(self):
        return self._step * self.dt

    def connect_reset(self):
        self.conn_mat = np.zeros((self.n_variable, self.n_variable))

    def connect_by_layer(self, source, target, conn_type='FC', ei='exc'):

        source_idx, target_idx, n_source, n_target = self.source_target_idx(source, target)

        if ei == 'exc':
            conn = 1
        elif ei == 'inh':
            conn = -1

        if conn_type == 'FC':
            self.conn_mat[source_idx: source_idx + n_source, target_idx: target_idx + n_target] = conn
        elif conn_type == 'one-to-one':
            self.conn_mat[source_idx: source_idx + n_source, target_idx: target_idx + n_target] = np.identity(n_target) * conn

    def initialize_weight(self):

        np_weights = np.zeros(self.conn_mat.shape)
        for layer_i in range(self.num_layers - 1):
            pre_begin, pre_end, post_begin, post_end = self.pre_post_idx(layer_i + 1, layer_i + 2)

            np_weights[pre_begin:pre_end, post_begin:post_end] = np.random.normal(1.0, 0.3, (
                self.num_neurons[layer_i], self.num_neurons[layer_i + 1])) * self.conn_mat[pre_begin:pre_end, post_begin:post_end]
            np_weights[post_begin:post_end, pre_begin:pre_end] = np_weights[pre_begin:pre_end, post_begin:post_end].T

        self.w = tf.Variable(np_weights)
        self.w_init = tf.Variable(np_weights)

    def source_target_idx(self, source, target):

        source_idx = sum(self.num_neurons[:source - 1])
        target_idx = sum(self.num_neurons[:target - 1])

        n_source = self.num_neurons[source - 1]
        n_target = self.num_neurons[target - 1]

        return source_idx, target_idx, n_source, n_target

    def pre_post_idx(self, source, target):
        pre_begin, pre_end = (sum(self.num_neurons[:source-1]), sum(self.num_neurons[:source]))
        post_begin, post_end = (sum(self.num_neurons[:target-1]), sum(self.num_neurons[:target]))

        return pre_begin, pre_end, post_begin, post_end

    def record_pre_post(self):

        if self._step == int(self.T/self.dt) - int(self.l_time/self.dt):

            self.xtr_record = tf.Variable(tf.zeros(sum(self.num_neurons), dtype=tf.float64))

        elif self._step > int(self.T/self.dt) - int(self.l_time/self.dt):

            self.xtr_record.assign_add(self.x_tr)

    def weight_update(self, source, target, lr):

        # load indices of source and target layers
        pre_begin, pre_end, post_begin, post_end = self.pre_post_idx(source, target)

        # weight btw layer i and i+1
        w_l = tf.slice(self.w, [pre_begin, post_begin], [pre_end - pre_begin, post_end - post_begin])

        # take the mean of synaptic output
        self.xtr_record.assign(self.xtr_record / int(self.l_time / self.dt))

        # synpatic current estimate from layer i to i+1
        xtr_l = tf.reshape(tf.slice(self.xtr_record, [source - 1], [pre_end - pre_begin]), (pre_end - pre_begin, 1))
        # synpatic current estimate from layer i+1 to i
        xtr_nl = tf.reshape(tf.slice(self.xtr_record, [target - 1], [post_end - post_begin]), (post_end - post_begin, 1))

        # post-synaptic current from layer i to i+1
        pre_isyn = w_l * xtr_l
        # post-synaptic current from layer i+1 to i
        post_isyn = tf.transpose(w_l) * xtr_nl

        # weight changes
        dw = lr * pre_isyn * tf.transpose(post_isyn)
        # update weights
        self.w[pre_begin:pre_end, post_begin:post_end].assign(tf.add(w_l, dw))


# network parameters
num_layers = 3
num_neurons_per_layer = [5, 5, 100]

# create external input
ext_current = np.random.normal(1000.0, 500.0, (5,)) * 10 ** -12

# build network
adex_01 = AdEx_Layer(neuron_model_constants=AdEx,
                     n_layers=num_layers,
                     n_neuron=num_neurons_per_layer)

# connect layers
adex_01.connect_by_layer(1, 2, conn_type='one-to-one')
adex_01.connect_by_layer(2, 3)
adex_01.connect_by_layer(3, 2, ei='inh')

# initialize weights
adex_01.initialize_weight()

# simulate
sim_dur = 1000 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-3)  # ms
learning_window = 200 * 10 ** -3

adex_01(sim_dur, dt, learning_window, ext_current)

# update weights
adex_01.weight_update(2, 3, 0.05)
