from AdEx_const import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
# from create_33images import create_shapes, plot_imgset
# from test_func import weight_dist
from mnist_data import create_mnist_set, plot_mnist_set, scale_tensor
from datetime import timedelta
from scipy import stats
import pickle5 as pickle

# A basic adex LIF neuron
class AdEx_Layer(object):

    def __init__(self,
                 num_pc_layers, num_pred_neurons,
                 num_stim,
                 gist_num, gist_connp, gist_maxw):
        """
        :param neuron_model_constants: dict. contains parameters of AdEx neuron.
        :param num_pc_layers: int. number of pc_layers.
        :param num_pred_neurons: list of int. number of prediction layers
        :param num_stim: int. size of stimulus.
        :param gist_num: int. size of gist.
        :param gist_connp: list of float.
        :param gist_maxw: list of int.
        """

        # for key in neuron_model_constants:
        #     setattr(self, key, neuron_model_constants[key])

        # network architecture
        self.n_pc_layer = num_pc_layers
        self.n_pred = num_pred_neurons
        self.n_gist = gist_num
        self.n_stim = num_stim

        # self.n_groups = num_pc_layers * 3 + 1
        self.neurons_per_group = [num_stim] + num_pred_neurons + [self.n_gist]

        self.n_variable = sum(self.neurons_per_group)

        # initial weight preparation
        self.w = {}
        self.w_init = {}
        self.connect_gist(conn_p=gist_connp, max_vals=gist_maxw)

        # constant weight
        # self.w_const = 1 * 10 ** -12
        # weight update time interval
        self.l_time = None

        # offset/bg current
        # self.offset = 600 * 10 ** -12

        # self.initialize_var()

    def initialize_var(self):

        # internal variables
        self.v = tf.Variable(tf.ones([self.n_variable, self.batch_size], dtype=tf.float32) * EL)
        self.c = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.ref = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        # pre-synaptic variables
        self.x = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.x_tr = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        # post-synaptic variable
        self.Isyn = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.fired = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.bool))

        self.xtr_record = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))

    def __call__(self, sim_duration, time_step, lt, I_ext, batch_size):

        # simulation parameters
        self.T = sim_duration
        self.dt = time_step

        self._step = 0
        self.l_time = lt

        self.batch_size = batch_size

        # initialize internal variables
        self.initialize_var()

        # feed external corrent to the first layer
        self.Iext = tf.constant(I_ext, dtype=tf.float32)

        for t in range(int(self.T / self.dt)):
            # update internal variables (v, c, x, x_tr)
            self.update_var()
            self.record_pre_post()

            self._step += 1

        # self.fr.assign(self.fr / self._step)
        # take the mean of synaptic output
        self.xtr_record.assign(self.xtr_record / int(self.l_time / self.dt))
        # self.fr = np.asarray(self.fr)

    def update_var(self):

        # feed synaptic current to higher layers
        self.update_Isyn()

        # current refractory status [0,2]
        # ref_constraint = tf.greater(self.ref, 0)
        ref_constraint = tf.cast(tf.greater(self.ref, 0), tf.float32)
        # update v according to ref: if in ref, dv = 0
        self.update_v(ref_constraint)
        self.update_c(ref_constraint)

        # subtract one time step (1) from refractory vector
        self.ref = tf.cast(tf.maximum(tf.subtract(self.ref, 1), 0), tf.float32)

        # update synaptic current
        self.update_x()
        self.update_xtr()

        # update spike monitor (fired: dtype=bool): if fired = True, else = False
        # self.fired = tf.greater_equal(self.v, self.VT)
        self.fired = tf.cast(tf.greater_equal(self.v, VT), tf.float32)

        # reset variables
        self.v = self.fired * EL + (1-self.fired) * self.v
        self.c = self.fired * tf.add(self.c, b) + (1-self.fired) * self.c
        self.x = self.fired * -x_reset + (1-self.fired) * self.x

        # set lower boundary of v (Vrest = -70.6 mV)
        self.v = tf.maximum(EL, self.v)
        self.ref = tf.add(self.ref, self.fired * float(t_ref / self.dt))

    def update_v(self, constraint):
        dv = (self.dt / Cm) * (gL * (EL - self.v) +
                                    gL * DeltaT * tf.exp((self.v - VT) / DeltaT) +
                                    self.Isyn - self.c)
        # dv_ref = tf.where(constraint, 0.0, dv)
        dv_ref = (1 -constraint) * dv
        self.v = tf.add(self.v, dv_ref)

    def update_c(self, constraint):
        dc = (self.dt / tauw) * (a * (self.v - EL) - self.c)
        # dc_ref = tf.where(constraint, 0, dc)
        dc_ref = (1 - constraint) * dc
        self.c = tf.add(self.c, dc_ref)

    def update_x(self):
        dx = self.dt * (-self.x / tau_rise)
        self.x = tf.add(self.x, dx)

    def update_xtr(self):
        dxtr = self.dt * (-self.x / tau_rise - self.x_tr / tau_s)
        self.x_tr = tf.add(self.x_tr, dxtr)

    def update_Isyn(self):

        record_pp = tf.cast(self._step > int(self.T / self.dt) - int(self.l_time / self.dt), dtype=tf.float32)

        # I = ext
        self.Isyn[:self.neurons_per_group[0]].assign(self.Iext)

        # gist = w['ig']Isyn['I']
        input_gist = tf.transpose(self.w['ig']) @ (self.x_tr[:self.neurons_per_group[0]])
        self.Isyn[-self.n_gist:, :].assign(input_gist)# + self.offset)

        ### write a function for pc layers
        # for layer_i, nums in enumerate(self.neurons_per_group):
        for pc_layer_idx in range(self.n_pc_layer):
            # within connections
            next_p_idx = sum(self.neurons_per_group[:pc_layer_idx + 1])
            next_p_size = self.n_pred[pc_layer_idx]

            gist = tf.transpose(self.w['gp' + str(pc_layer_idx + 1)]) @ (self.x_tr[-self.n_gist:, :])

            self.Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(gist)# + self.offset)

    def connect_gist(self, conn_p, max_vals):
        '''
        :param conn_p: list of float. connection probabilities ranges between [0,1]
        :param max_vals: list of int. max weight values.
        :return: w['ig'] and w['gp']
        '''
        # needs to be shortened!!!
        ig_shape = (self.neurons_per_group[0], self.n_gist)
        # rand_w = tf.random.normal(shape=ig_shape, mean=max_vals[0], stddev=max_vals[0]/10.0, dtype=tf.float32)
        rand_w = tf.random.normal(shape=ig_shape, mean=max_vals[0], stddev=max_vals[0]/2.0, dtype=tf.float32)
        # rand_w = tf.random.uniform(shape=ig_shape, minval=0.0, maxval=max_vals[0], dtype=tf.float32)
        constraint = tf.cast(tf.greater(tf.random.uniform(shape=ig_shape), 1 - conn_p[0]), tf.float32)
        self.w['ig'] = constraint * rand_w

        for pc_layer_idx in range(self.n_pc_layer):

            gp_shape = (self.n_gist, self.n_pred[pc_layer_idx])
            # rand_w = tf.random.normal(shape=gp_shape, mean=max_vals[pc_layer_idx+1], stddev=max_vals[pc_layer_idx+1]/10.0, dtype=tf.float32)
            # rand_w = tf.random.normal(shape=gp_shape, mean=max_vals[pc_layer_idx+1], stddev=max_vals[pc_layer_idx+1]/10.0, dtype=tf.float32)
            rand_w = tf.abs(tf.random.normal(shape=gp_shape,
                                      mean=max_vals[pc_layer_idx + 1],
                                      stddev=max_vals[pc_layer_idx + 1] / 2.0,
                                      dtype=tf.float32))
            # rand_w = tf.random.uniform(shape=gp_shape, minval=0.0, maxval=max_vals[pc_layer_idx+1], dtype=tf.float32)
            constraint = tf.cast(tf.greater(tf.random.uniform(shape=gp_shape), 1 - conn_p[pc_layer_idx+1]), tf.float32)
            self.w['gp' + str(pc_layer_idx + 1)] = constraint * rand_w

    def record_pre_post(self):

        if self._step == int(self.T / self.dt) - int(self.l_time / self.dt):
            self.xtr_record = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))

        elif self._step > int(self.T / self.dt) - int(self.l_time / self.dt):
            # self.xtr_record[range1, range2].assign_add(sum_syn)
            # self.xtr_record.assign_add(self.Isyn)
            self.xtr_record.assign_add(self.x_tr)# + self.offset)

    def train_network(self, num_epoch, sim_dur, sim_dt, sim_lt,
                      input_current,
                      n_shape, batch_size):
        """
        :param num_epoch:
        :param sim_dur:
        :param sim_dt:
        :param sim_lt:
        :param input_current:
        :param n_shape:
        :param batch_size:
        :return:
        """

        n_batch = int(input_current.shape[1] / batch_size)

        start_time = time.time()
        sse = []
        epoch_time_avg = 0

        for epoch_i in range(num_epoch):

            epoch_time = time.time()

            for iter_i in range(n_batch):

                iter_time = time.time()
                curr_batch = input_current[:, iter_i * batch_size:(iter_i + 1) * batch_size]


                self.__call__(sim_duration=sim_dur, time_step=sim_dt, lt=sim_lt,
                              I_ext=curr_batch,
                              batch_size=batch_size)


                end_iter_time = time.time()
                print('epoch #{0}/{1} = {2:.2f}, iter #{3}/{4} = {5:.2f} sec'.format(epoch_i + 1, num_epoch,
                                                                                     end_iter_time - epoch_time,
                                                                                     iter_i + 1, n_batch,
                                                                                     end_iter_time - iter_time))

            fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(4 * 5, 4 * 1))
            # plot progres
            input_rdm = self.matrix_rdm((self.xtr_record[:adex_01.n_stim, :]).numpy().T)

            g_img = self.xtr_record[-adex_01.n_gist:].numpy().T
            # g_img = adex_01.Isyn[-adex_01.n_gist:].numpy().T
            g_rdm = self.matrix_rdm(g_img)

            p1_img = self.xtr_record[sum(adex_01.neurons_per_group[:1]):sum(adex_01.neurons_per_group[:2])].numpy().T
            # p1_img = adex_01.Isyn[784 : 784 + adex_01.n_pred[0], :].numpy().T
            p1_rdm = self.matrix_rdm(p1_img)

            p2_img = self.xtr_record[sum(adex_01.neurons_per_group[:2]):sum(adex_01.neurons_per_group[:3])].numpy().T
            # p2_img = adex_01.Isyn[784 + adex_01.n_pred[0] : 784 + adex_01.n_pred[0] + adex_01.n_pred[1], :].numpy().T
            p2_rdm = self.matrix_rdm(p2_img)

            inp_plot = axs[0].imshow(input_rdm, cmap='Reds', vmin=0, vmax=1)
            fig.colorbar(inp_plot, ax=axs[0], shrink=0.6)
            g_plot = axs[1].imshow(g_rdm, cmap='Reds')#, vmin=0, vmax=1)
            fig.colorbar(g_plot, ax=axs[1], shrink=0.6)
            p1_plot = axs[2].imshow(p1_rdm, cmap='Reds')#, vmin=-0, vmax=1)
            fig.colorbar(p1_plot, ax=axs[2], shrink=0.6)
            p2_plot = axs[3].imshow(p2_rdm, cmap='Reds')#, vmin=-0, vmax=1)
            fig.colorbar(p2_plot, ax=axs[3], shrink=0.6)
            fig.suptitle('progress update: epoch #{0}/{1}'.format(epoch_i + 1, num_epoch))
            plt.show()

            # time remaining
            epoch_time_avg += time.time() - epoch_time
            print('***** time remaining = {0}'.format(
                str(timedelta(seconds=epoch_time_avg / (epoch_i + 1) * (num_epoch - epoch_i - 1)))))

        end_time = time.time()
        print('simulation : {0:.2f} sec'.format(end_time - start_time))

        return fig

    def matrix_rdm(self, matrix_data):
        output = np.array([1 - stats.spearmanr(matrix_data[ni], matrix_data[mi])[0]
                           for ni in range(len(matrix_data))
                           for mi in range(len(matrix_data))]).reshape(len(matrix_data), len(matrix_data))

        return output

# network parameters
n_pred_neurons = [30**2, 25**2, 20**2]
n_pc_layers = len(n_pred_neurons)
n_gist = 144

# create external input
batch_size = 30
n_shape = 3
n_samples = 10

# ext_current, digits, test_set_idx, label_set_shuffled = create_mnist_set(nDigit=n_shape, nSample=n_samples, shuffle=False)
ext_current, label_set_shuffled, test_set, test_labels, classes, test_set_idx = create_mnist_set(data_type=tf.keras.datasets.mnist,
                                                                                                   nDigit=n_shape,
                                                                                                   nSample=n_samples,
                                                                                                   shuffle=True)
# ext_current *= pamp
n_stim = ext_current.shape[1]
sqrt_nstim = int(np.sqrt(n_stim))



# plot the same test set
plot_mnist_set(testset=ext_current, testset_idx=test_set_idx, nDigit=n_shape, nSample=n_samples, savefolder='test_figures')
plt.show()

# def conn_probs(n_a, n_b):
#     return np.sqrt(n_b/n_a) * 0.2 #* 0.025
#
# conn_vals = np.array([conn_probs(a_i, b_i)
#                       for a_i, b_i in zip([n_stim] + [n_gist] * n_pc_layers, [n_gist] + n_pred_neurons)])
# max_vals = np.array([1] * (n_pc_layers + 1))# * 0.2
conn_vals = [0.005] * (n_pc_layers + 1)
# max_vals  = [np.sqrt(1/mv_i) for mv_i in np.array([n_gist] + n_pred_neurons)]
max_vals = np.array([n_gist/n_stim] + [np.sqrt(n_pred_neurons[i] / n_gist) for i in range(n_pc_layers)])


# build network
adex_01 = AdEx_Layer(num_pc_layers=n_pc_layers,
                     num_pred_neurons=n_pred_neurons,
                     num_stim=n_stim,
                     gist_num=n_gist, gist_connp=conn_vals, gist_maxw=max_vals)

# simulate
sim_dur = 350 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-4)  # ms
learning_window = 100 * 10 ** -3

n_epoch = 1

#     def train_network(self, num_epoch, sim_dur, sim_dt, sim_lt, input_current, n_shape, batch_size):
rdm_fig = adex_01.train_network(num_epoch=n_epoch, sim_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                                input_current=ext_current.T, n_shape=n_shape, batch_size=batch_size)


aaa = adex_01.xtr_record[-adex_01.n_gist:, ::10].numpy()/pamp
bbb = adex_01.xtr_record[sum(adex_01.neurons_per_group[:1]):sum(adex_01.neurons_per_group[:2]), ::10].numpy()/pamp
ccc = adex_01.xtr_record[sum(adex_01.neurons_per_group[:2]):sum(adex_01.neurons_per_group[:3]), ::10].numpy()/pamp

fig, axs = plt.subplots(ncols=3, nrows=3)
# fig2, axs2 = plt.subplots(ncols=2, nrows=3)
for i in range(3):
    a1 = axs[0, i].imshow(aaa[:, i].reshape(
        int(np.sqrt(n_gist)), int(np.sqrt(n_gist))),
        cmap='Reds')#, vmin=600, vmax=3000)
    fig.colorbar(a1, ax=axs[0,i], shrink=0.5)
    a2 = axs[1, i].imshow(bbb[:, i].reshape(
        int(np.sqrt(n_pred_neurons[0])), int(np.sqrt(n_pred_neurons[0]))),
        cmap='Reds')#, vmin=600, vmax=3000)
    fig.colorbar(a2, ax=axs[1, i], shrink=0.5)
    a3 = axs[2, i].imshow(ccc[:, i].reshape(
        int(np.sqrt(n_pred_neurons[1])), int(np.sqrt(n_pred_neurons[1]))),
        cmap='Reds')#, vmin=600, vmax=3000)
    fig.colorbar(a3, ax=axs[2, i], shrink=0.5)
plt.show()