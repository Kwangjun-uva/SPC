import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# List all your physical GPUs
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

    # Pick an physiological behaviour
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

    def __init__(self, neuron_model_constants, n_pc_layers, n_pred_neurons, n_stim):  # n_layers, n_neuron):
        """
        :param neuron_model_constants: dict
        :param n_pc_layers: int
        :param n_pred_neurons: list
        """

        for key in neuron_model_constants:
            setattr(self, key, neuron_model_constants[key])

        # network architecture
        # self.num_layers = n_layers
        # self.num_neurons = n_neuron
        # self.n_pc_layers = n_pc_layers

        self.n_groups = n_pc_layers * 3 + 1
        self.neurons_per_group = [n_stim] * 3 + np.repeat([n_pred_neurons[:-1]], 3).tolist() + [n_pred_neurons[-1]]

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
        self.np_weights = np.zeros(self.conn_mat.shape)

        # constant weight
        self.w_const = 550 * 10 ** -12
        # self.w_const = 600 * 10 ** -12

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

        # self.fs = tf.Variable(tf.zeros(sum(self.num_neurons), dtype=tf.int64))
        # self.fr = tf.Variable(tf.zeros([self.n_variable, ], dtype=tf.float64))
        # self.x_trace = []
        self.fr = []

        for t in range(int(self.T / self.dt)):
            # update internal variables (v, c, x, x_tr)
            self.update_var()
            # update synaptic variable (Isyn = w * x_tr + Iext)
            # self.update_Isyn()
            self.record_pre_post()

            # save firing rate (fs) and firing time (fr)
            fired_float = tf.cast(self.fired, dtype=tf.float64)
            self.fr.append(fired_float)
            # self.fr.assign_add(fired_float)

            # self.x_trace.append(self.x_tr[9:27].numpy())

            self._step += 1

        # self.fr.assign(self.fr / self._step)
        # take the mean of synaptic output
        self.xtr_record.assign(self.xtr_record / int(self.l_time / self.dt))
        # self.x_trace.assign(self.xtr_record)

    def create_Iext(self, Iext):

        Iext_np = np.zeros(self.n_variable)
        # Iext_np[:self.num_neurons[0]] = Iext
        Iext_np[:self.neurons_per_group[0]] = Iext

        return tf.constant(Iext_np)

    def update_var(self):

        # feed synaptic current to higher layers
        self.Isyn = self.update_Isyn()

        # current refractory status [0,2]
        ref_constraint = tf.greater(self.ref, 0)
        # update v according to ref: if in ref, dv = 0
        self.update_v(ref_constraint)
        self.update_c(ref_constraint)

        # subtract one time step (1) from refractory vector
        self.ref = tf.maximum(tf.subtract(self.ref, 1), 0)

        # update synaptic current
        self.update_x()
        self.update_xtr()

        # update spike monitor (fired: dtype=bool): if fired = True, else = False
        self.fired = tf.greater_equal(self.v, self.VT)
        # reset variables
        self.v = tf.where(self.fired, self.EL, self.v)
        self.c = tf.where(self.fired, tf.add(self.c, self.b), self.c)
        self.x = tf.where(self.fired, -self.x_reset, self.x)

        # set lower boundary of v (Vrest = -70.6 mV)
        self.v = tf.maximum(self.EL, self.v)
        # # if in refractory period, set v to Vrest
        # self.v = tf.where(ref_constraint, self.EL, self.v)

        # self.x = tf.where(self.fired, -self.x_reset, self.x)
        # update refractory vector : if fired = 2, else = 0
        self.ref = tf.add(self.ref, tf.where(self.fired, int(self.t_ref / self.dt), 0))

    def update_v(self, constraint):
        dv = (self.dt / self.Cm) * (self.gL * (self.EL - self.v) +
                                    self.gL * self.DeltaT * tf.exp((self.v - self.VT) / self.DeltaT) +
                                    self.Isyn - self.c)
        dv_ref = tf.where(constraint, 0.0, dv)
        self.v = tf.add(self.v, dv_ref)

    def update_c(self, constraint):
        dc = (self.dt / self.tauw) * (self.a * (self.v - self.EL) - self.c)
        dc_ref = tf.where(constraint, 0, dc)
        self.c = tf.add(self.c, dc_ref)

    def update_x(self):
        dx = self.dt * (-self.x / self.tau_rise)
        self.x = tf.add(self.x, dx)

    def update_xtr(self):
        dxtr = self.dt * (-self.x / self.tau_rise - self.x_tr / self.tau_s)
        self.x_tr = tf.add(self.x_tr, dxtr)

    def update_Isyn(self):
        # return tf.tensordot(tf.transpose(self.w * self.conn_mat), self.x_tr, 1) + self.Iext
        return tf.tensordot(tf.transpose(self.w), self.x_tr * self.w_const, 1) + self.Iext

    def get_current_timestep(self):
        return self._step * self.dt

    def connect_reset(self):
        self.conn_mat = np.zeros((self.n_variable, self.n_variable))

    def connect_by_neurongroup(self, source, target, conn_type='FC', syn_type='exc', constant=False):

        # source_idx, target_idx, n_source, n_target = self.source_target_idx(source, target)
        source_begin, source_end, target_begin, target_end = self.pre_post_idx(source, target)

        syn_val = 1
        if syn_type == 'inh':
            syn_val = -1

        if conn_type == 'FC':
            # self.conn_mat[source_idx: source_idx + n_source, target_idx: target_idx + n_target] = syn_val
            self.conn_mat[source_begin: source_end, target_begin: target_end] = syn_val

        elif conn_type == 'one-to-one':
            # self.conn_mat[source_idx: source_idx + n_source,
            # target_idx: target_idx + n_target] = np.identity(n_target) * syn_val
            one_to_one_conn = np.identity(target_end - target_begin) * syn_val
            self.conn_mat[source_begin: source_end, target_begin: target_end] = one_to_one_conn
            if constant == True:
                self.np_weights[source_begin:source_end, target_begin:target_end] = one_to_one_conn

    def randomize_weights(self, source, target, symmetric=False, sym_syn_type='exc', target_w=None,
                          target_sym_syn_type='inh'):

        source_begin, source_end, target_begin, target_end = self.pre_post_idx(source, target)
        n_target = (target_end - target_begin) * 0.4
        # n_target = 5
        rand_weights = np.random.normal(1.5, 0.8, (
        self.neurons_per_group[source - 1], self.neurons_per_group[target - 1])) / n_target * self.conn_mat[
                                                                                              source_begin:source_end,
                                                                                              target_begin:target_end]
        self.np_weights[source_begin:source_end, target_begin:target_end] = rand_weights

        if symmetric:
            self.symmetric_weight(rand_weights, source_begin, source_end, target_begin, target_end, sym_syn_type)

        if target_w:
            w_source_begin, w_source_end, w_target_begin, w_target_end = self.pre_post_idx(*target_w)
            self.np_weights[w_source_begin:w_source_end, w_target_begin:w_target_end] = rand_weights * -1
            self.symmetric_weight(rand_weights, w_source_begin, w_source_end, w_target_begin, w_target_end,
                                  target_sym_syn_type)

    def symmetric_weight(self, w_mat, source_begin, source_end, target_begin, target_end, syn_type):
        w_tranpose = w_mat.T
        if syn_type == 'inh':
            ei_val = -1
        else:
            ei_val = 1
        self.np_weights[target_begin:target_end, source_begin:source_end] = w_tranpose * ei_val

    def initialize_weight(self):

        self.w = tf.Variable(self.np_weights)
        self.w_init = tf.constant(self.np_weights)

        del self.np_weights

    def pre_post_idx(self, source, target):
        pre_begin, pre_end = (sum(self.neurons_per_group[:source - 1]), sum(self.neurons_per_group[:source]))
        post_begin, post_end = (sum(self.neurons_per_group[:target - 1]), sum(self.neurons_per_group[:target]))

        return pre_begin, pre_end, post_begin, post_end

    def record_pre_post(self):

        if self._step == int(self.T / self.dt) - int(self.l_time / self.dt):

            self.xtr_record = tf.Variable(tf.zeros(self.n_variable, dtype=tf.float64))

        elif self._step > int(self.T / self.dt) - int(self.l_time / self.dt):

            self.xtr_record.assign_add(self.x_tr * self.w_const)

    def hebbian_dw(self, source, target, lr, reg_alpha):

        # load indices of source and target layers
        pre_begin, pre_end, post_begin, post_end = self.pre_post_idx(source, target)

        # weight btw layer i and i+1
        # w_l = tf.slice(self.w, [pre_begin, post_begin], [pre_end - pre_begin, post_end - post_begin])
        w_l = tf.slice(self.w, [pre_begin, post_begin], [pre_end - pre_begin, post_end - post_begin])

        # # take the mean of synaptic output
        # self.xtr_record.assign(self.xtr_record / int(self.l_time / self.dt))

        # synaptic current estimate from layer i to i+1
        xtr_l = tf.reshape(tf.slice(self.xtr_record, [sum(self.neurons_per_group[:source - 1])], [pre_end - pre_begin]), (pre_end - pre_begin, 1))
        # synaptic current estimate from layer i+1 to i
        xtr_nl = tf.reshape(tf.slice(self.xtr_record, [sum(self.neurons_per_group[:target - 1])], [post_end - post_begin]),
                            (post_end - post_begin, 1))

        # post-synaptic current from layer i to i+1
        pre_isyn = tf.abs(w_l) * xtr_l / 10 ** -12
        # post-synaptic current from layer i+1 to i
        post_isyn = tf.transpose(tf.abs(w_l)) * xtr_nl / 10 ** -12

        # weight changes
        dw = lr * pre_isyn * tf.transpose(post_isyn) - 2 * reg_alpha * tf.abs(w_l)
        # update weights
        dw_sign = self.conn_mat[pre_begin, post_begin]

        return dw * dw_sign
        # self.w[pre_begin:pre_end, post_begin:post_end].assign(tf.add(w_l, dw * dw_sign))

    def weight_update(self, positive_w, negative_w, dw_tensor):
        pre_begin, pre_end, post_begin, post_end = self.pre_post_idx(*positive_w)
        w_l = tf.slice(self.w, [pre_begin, post_begin], [pre_end - pre_begin, post_end - post_begin])
        new_weights = tf.maximum(tf.add(w_l, dw_tensor), 0.0)

        # update weights to E+ group
        self.w[pre_begin:pre_end, post_begin:post_end].assign(new_weights)
        self.w[post_begin:post_end, pre_begin:pre_end].assign(tf.transpose(new_weights) * -1)

        # update the same weights to E- group
        pre_begin, pre_end, post_begin, post_end = self.pre_post_idx(*negative_w)
        self.w[pre_begin:pre_end, post_begin:post_end].assign(new_weights * -1)
        self.w[post_begin:post_end, pre_begin:pre_end].assign(tf.transpose(new_weights))

pamp = 10 ** -12

# network parameters
n_pc_layers = 1
n_pred_neurons = [25]

# create external input
n_stim = 4
ext_current = np.random.normal(2000.0, 200.0, (n_stim,)) * 10 ** -12

start_time = time.time()
# build network
adex_01 = AdEx_Layer(neuron_model_constants=AdEx,
                     n_pc_layers=n_pc_layers,
                     n_pred_neurons=n_pred_neurons,
                     n_stim=n_stim)

# connect layers
# Input => E0+
adex_01.connect_by_neurongroup(1, 2, conn_type='one-to-one', syn_type='exc', constant=True)
# Input => E0-
adex_01.connect_by_neurongroup(1, 3, conn_type='one-to-one', syn_type='inh', constant=True)
# E0+ <=> P1
adex_01.connect_by_neurongroup(2, 4, syn_type='exc')
adex_01.connect_by_neurongroup(4, 2, syn_type='inh')
# E0- <=> P1
adex_01.connect_by_neurongroup(3, 4, syn_type='inh')
adex_01.connect_by_neurongroup(4, 3, syn_type='exc')

# initialize weights
adex_01.randomize_weights(2, 4, symmetric=True, sym_syn_type='inh', target_w=[3, 4], target_sym_syn_type='exc')

# convert weights to tensors and save the initial weights
adex_01.initialize_weight()
build_end_time = time.time()

# simulate
sim_dur = 500 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-4)  # ms
learning_window = 200 * 10 ** -3

iter_n = 1
sse = []

for iter_i in range(iter_n):

    iter_time = time.time()
    adex_01(sim_dur, dt, learning_window, ext_current)

    # update weights
    # compute gradient
    dw_p = adex_01.hebbian_dw(2, 4, 0.0000015, 0.00005)
    dw_n = adex_01.hebbian_dw(3, 4, 0.0000015, 0.00005)
    # update weight
    adex_01.weight_update([2, 4], [3, 4], tf.add(dw_p, dw_n))

    end_iter_time = time.time()
    print('iter# {0} took {1:.2f} sec'.format(iter_i + 1, end_iter_time - iter_time))

    sqrt_nstim = int(np.sqrt(n_stim))
    original_image = tf.reshape(adex_01.Iext[:n_stim], (sqrt_nstim, sqrt_nstim)) / pamp
    input_image = tf.reshape(adex_01.xtr_record[:n_stim], (sqrt_nstim, sqrt_nstim)) / pamp
    reconstructed_image = tf.reshape(tf.tensordot(tf.transpose(adex_01.w[n_stim*3:, n_stim*2:n_stim*3]), adex_01.xtr_record[n_stim*3:], 1),
                                     (sqrt_nstim, sqrt_nstim)) / pamp
    sse.append(np.sum(np.abs(tf.subtract(reconstructed_image, input_image).numpy())))
    print (reconstructed_image)

    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    original_plot = plt.imshow(original_image, cmap='Reds', vmin=1000, vmax=4000)
    plt.colorbar(original_plot, shrink=0.6)
    plt.subplot(142)
    input_plot = plt.imshow(input_image, cmap='Reds', vmin=1000, vmax=4000)
    plt.colorbar(input_plot, shrink=0.6)
    plt.subplot(143)
    reconst_plot = plt.imshow(reconstructed_image, cmap='Reds', vmin=1000, vmax=4000)
    plt.colorbar(reconst_plot, shrink=0.6)
    plt.subplot(144)
    diff_plot = plt.imshow(input_image - reconstructed_image, cmap='bwr', vmin=-1000, vmax=1000)
    plt.colorbar(diff_plot, shrink=0.6)
    plt.show()

plt.figure(figsize=(5,5))
plt.plot(sse)
plt.show()

# input to output plot (given Iext, what is EPSC?)
# plt.scatter(adex_01.Isyn[:n_stim]/pamp, adex_01.xtr_record[:n_stim]/pamp)
# plt.plot(adex_01.Isyn[:n_stim]/pamp, adex_01.Isyn[:n_stim]/pamp, c='r', ls='--')
# plt.show()

end_time = time.time()
print('building : {0:.2f} sec\nsimulation : {1:.2f} sec\ntotal : {2:.2f} sec'.format(build_end_time - start_time,
                                                                                     end_time - build_end_time,
                                                                                     end_time - start_time))


# plt.show()
