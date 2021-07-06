import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
# from create_33images import create_shapes, plot_imgset
from test_func import weight_dist
from mnist_data import create_mnist_set, plot_mnist_set
from datetime import timedelta

# # List all your physical GPUs
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

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

    def __init__(self, neuron_model_constants, num_pc_layers, num_pred_neurons, num_stim, num_gist):
        """
        :param neuron_model_constants: dict
        :param num_pc_layers: int
        :param num_pred_neurons: list
        """

        for key in neuron_model_constants:
            setattr(self, key, neuron_model_constants[key])

        # network architecture
        self.n_pc_layer = num_pc_layers
        self.n_pred = num_pred_neurons
        self.n_gist = num_gist

        # self.n_groups = num_pc_layers * 3 + 1
        self.neurons_per_group = [num_stim] * 3 + np.repeat([num_pred_neurons[:-1]], 3).tolist() + \
                                 [num_pred_neurons[-1]]

        self.n_variable = sum(self.neurons_per_group)

        # initial weight preparation
        self.w = {}
        self.w_init = {}
        self.connect_pc()

        # constant weight
        self.w_const = 550 * 10 ** -12
        # weight update time interval
        self.l_time = None

        self.xtr_record = None

        # self.initialize_var()

    def initialize_var(self):

        # internal variables
        self.v = tf.Variable(tf.ones([self.n_variable, self.batch_size], dtype=tf.float32) * self.EL)
        self.c = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.ref = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.int32))
        # pre-synaptic variables
        self.x = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.x_tr = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        # post-synaptic variable
        self.Isyn = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.fired = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.bool))

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
        # self.Iext = self.create_Iext(I_ext)
        self.Iext = tf.constant(I_ext, dtype=tf.float32)

        # self.fr = []

        for t in range(int(self.T / self.dt)):
            # update internal variables (v, c, x, x_tr)
            self.update_var()
            # update synaptic variable (Isyn = w * x_tr + Iext)
            # self.update_Isyn()
            self.record_pre_post()

            # save firing rate (fs) and firing time (fr)
            # fired_float = tf.cast(self.fired, dtype=tf.float32)
            # self.fr.append(fired_float)
            # self.fr.assign_add(fired_float)

            self._step += 1

        # self.fr.assign(self.fr / self._step)
        # take the mean of synaptic output
        self.xtr_record.assign(self.xtr_record / int(self.l_time / self.dt))
        # self.fr = np.asarray(self.fr)

    def update_var(self):

        # feed synaptic current to higher layers
        self.update_Isyn()

        # current refractory status [0,2]
        ref_constraint = tf.greater(self.ref, 0)
        # update v according to ref: if in ref, dv = 0
        self.update_v(ref_constraint)
        self.update_c(ref_constraint)

        # subtract one time step (1) from refractory vector
        self.ref = tf.maximum(tf.subtract(self.ref, int((1 * 10 ** -1) / self.dt)), 0)

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

        # I = ext
        self.Isyn[:self.neurons_per_group[0]].assign(self.Iext)

        ### write a function for pc layers
        # for layer_i, nums in enumerate(self.neurons_per_group):
        for pc_layer_idx in range(self.n_pc_layer):
            # within connections
            curr_p_idx = sum(self.neurons_per_group[:pc_layer_idx * 3])
            curr_p_size = self.neurons_per_group[pc_layer_idx * 3]

            next_p_idx = sum(self.neurons_per_group[:pc_layer_idx * 3 + 3])
            next_p_size = self.neurons_per_group[pc_layer_idx * 3 + 3]

            bu_sensory = self.x_tr[curr_p_idx: curr_p_idx + curr_p_size, :] * self.w_const
            td_pred = self.w['pc' + str(pc_layer_idx + 1)] @ (
                    self.x_tr[next_p_idx:next_p_idx + next_p_size, :] * self.w_const)

            # E+ = I - P
            self.Isyn[curr_p_idx + curr_p_size:curr_p_idx + 2 * curr_p_size, :].assign(tf.add(bu_sensory, -td_pred))
            # E- = -I + P
            self.Isyn[curr_p_idx + 2 * curr_p_size:next_p_idx, :].assign(tf.add(-bu_sensory, td_pred))

            # P = bu_error + td_error
            bu_err_pos = tf.transpose(self.w['pc' + str(pc_layer_idx + 1)]) @ (
                    self.x_tr[curr_p_idx + curr_p_size:curr_p_idx + 2 * curr_p_size, :] * self.w_const)
            bu_err_neg = tf.transpose(self.w['pc' + str(pc_layer_idx + 1)]) @ (
                    self.x_tr[curr_p_idx + 2 * curr_p_size:next_p_idx, :] * self.w_const)

            if pc_layer_idx < self.n_pc_layer - 1:
                td_err_pos = self.x_tr[next_p_idx + next_p_size:next_p_idx + 2 * next_p_size] * self.w_const
                td_err_neg = self.x_tr[next_p_idx + 2 * next_p_size:next_p_idx + 3 * next_p_size] * self.w_const
                self.Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(tf.add(tf.add(bu_err_pos, -bu_err_neg),
                                                                                tf.add(-td_err_pos, td_err_neg)))
            else:
                self.Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(tf.add(bu_err_pos, -bu_err_neg))

    def connect_pc(self):

        for pc_layer_idx in range(self.n_pc_layer):
            err_size = self.neurons_per_group[pc_layer_idx * 3 + 1]
            pred_size = self.n_pred[pc_layer_idx]

            norm_factor = 0.1 * pred_size

            self.w['pc' + str(pc_layer_idx + 1)] = tf.random.normal((err_size, pred_size), 1.0, 0.3) / norm_factor
            self.w_init['pc' + str(pc_layer_idx + 1)] = self.w['pc' + str(pc_layer_idx + 1)]

    def connect_gist(self, conn_p, max_vals):

        self.w['ig'] = tf.random.normal(shape=(self.n_gist),
                                        mean=max_vals[0], stddev=max_vals[0]/10,
                                        dtype=tf.float32)
        for pc_layer_idx in range(self.n_pc_layer):
            self.w['gp' + str(pc_layer_idx + 1)] =



    def record_pre_post(self):

        if self._step == int(self.T / self.dt) - int(self.l_time / self.dt):

            self.xtr_record = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))

        elif self._step > int(self.T / self.dt) - int(self.l_time / self.dt):

            self.xtr_record.assign_add(self.x_tr * self.w_const)

    # def hebbian_dw(self, source, target, lr, reg_alpha):
    def weight_update(self, lr, reg_alpha):

        for pc_layer_idx in range(self.n_pc_layer):
            err_idx = sum(self.neurons_per_group[:pc_layer_idx * 3 + 1])
            err_size = self.neurons_per_group[pc_layer_idx * 3 + 1]
            pred_idx = sum(self.neurons_per_group[:pc_layer_idx * 3 + 3])
            pred_size = self.neurons_per_group[pc_layer_idx * 3 + 3]

            xtr_ep = self.xtr_record[err_idx: err_idx + err_size]
            xtr_en = self.xtr_record[err_idx + err_size: err_idx + 2 * err_size]
            xtr_p = self.xtr_record[pred_idx: pred_idx + pred_size]

            dw_all_pos = lr * tf.einsum('ij,kj->ikj', xtr_ep / 10 ** -12, xtr_p / 10 ** -12)
            dw_all_neg = lr * tf.einsum('ij,kj->ikj', xtr_en / 10 ** -12, xtr_p / 10 ** -12)

            dw_mean_pos = tf.reduce_mean(dw_all_pos, axis=2) - 2 * reg_alpha * tf.abs(
                self.w['pc' + str(pc_layer_idx + 1)])
            dw_mean_neg = tf.reduce_mean(dw_all_neg, axis=2) - 2 * reg_alpha * tf.abs(
                self.w['pc' + str(pc_layer_idx + 1)])

            dws = tf.add(dw_mean_pos, -dw_mean_neg)

            self.w['pc' + str(pc_layer_idx + 1)] = tf.maximum(tf.add(self.w['pc' + str(pc_layer_idx + 1)], dws), 0.0)


    def test_inference(self, imgs, stim_shape, stim_type, sim_dur, sim_dt, sim_lt, digit_list=None):

        if stim_type == 'novel':
            test_current, digits, test_set_idx, label_set_shuffled = create_mnist_set(nDigit=stim_shape, nSample=1,
                                                                                      test_digits=digit_list)

        elif stim_type == 'trained':
            test_current = imgs[::int(imgs.shape[0] / stim_shape)]

        # load the model
        self.__call__(sim_duration=sim_dur, time_step=sim_dt, lt=sim_lt,
                      I_ext=test_current.T * 10 ** -12,
                      batch_size=test_current.shape[0])

        sqrt_nstim = int(np.sqrt(self.Iext.shape[0]))
        original_image = tf.reshape(self.Iext, (sqrt_nstim, sqrt_nstim, test_current.shape[0])) / pamp
        input_image = tf.reshape(self.xtr_record[:self.Iext.shape[0], :], (sqrt_nstim, sqrt_nstim, test_current.shape[0])) / pamp
        reconstructed_image = tf.reshape(
            self.w['pc1'] @ self.xtr_record[self.Iext.shape[0] * 3:self.Iext.shape[0] * 3 + self.n_pred[0], :],
            (sqrt_nstim, sqrt_nstim, test_current.shape[0])) / pamp

        fig, axs = plt.subplots(ncols=4, nrows=test_current.shape[0], figsize=(4 * 5, 4 * test_current.shape[0]))
        for shape_i in range(test_current.shape[0]):
            original_plot = axs[shape_i, 0].imshow(original_image[:, :, shape_i], cmap='Reds', vmin=1000, vmax=4000)
            fig.colorbar(original_plot, ax=axs[shape_i, 0], shrink=0.6)
            input_plot = axs[shape_i, 1].imshow(input_image[:, :, shape_i], cmap='Reds', vmin=1000, vmax=4000)
            fig.colorbar(input_plot, ax=axs[shape_i, 1], shrink=0.6)
            reconst_plot = axs[shape_i, 2].imshow(reconstructed_image[:, :, shape_i], cmap='Reds', vmin=1000, vmax=4000)
            fig.colorbar(reconst_plot, ax=axs[shape_i, 2], shrink=0.6)
            diff_plot = axs[shape_i, 3].imshow(input_image[:, :, shape_i] - reconstructed_image[:, :, shape_i], cmap='bwr',
                                               vmin=-1000, vmax=1000)
            fig.colorbar(diff_plot, ax=axs[shape_i, 3], shrink=0.6)

        return fig


    def train_network(self, num_epoch, sim_dur, sim_dt, sim_lt,
                      lr, reg_a,
                      input_current,
                      n_shape, batch_size,
                      set_idx):

        n_batch = int(input_current.shape[1] / batch_size)
        shape_size = int(input_current.shape[1] / n_shape)

        start_time = time.time()
        sse = []
        epoch_time_avg = 0

        for epoch_i in range(num_epoch):

            epoch_time = time.time()

            fig, axs = plt.subplots(ncols=3, nrows=n_shape, figsize=(4 * 5, 4 * n_shape))

            plt_idx = 0
            for iter_i in range(n_batch):
                iter_time = time.time()
                curr_batch = input_current[:, iter_i * batch_size:(iter_i + 1) * batch_size]
                curr_idx = set_idx[iter_i * batch_size:(iter_i + 1) * batch_size]

                self.__call__(sim_duration=sim_dur, time_step=sim_dt, lt=sim_lt,
                              I_ext=curr_batch,
                              batch_size=batch_size)

                # update weightss
                self.weight_update(lr=lr, reg_alpha=reg_a)

                end_iter_time = time.time()
                print('epoch #{0}/{1} = {2:.2f}, iter #{3}/{4} = {5:.2f} sec'.format(epoch_i + 1, num_epoch,
                                                                                     end_iter_time - epoch_time,
                                                                                     iter_i + 1, n_batch,
                                                                                     end_iter_time - iter_time))

                if plt_idx < 3:
                    digit_idx = np.random.choice(np.argwhere((curr_idx > plt_idx * shape_size) &
                                                             (curr_idx < (plt_idx + 1) * shape_size)).T[0])
                else:
                    digit_idx = []

                print (digit_idx)
                if (epoch_i % 3 == 0) and digit_idx:
                    print (curr_idx)
                    print ('plt_idx = {0}, plot={1}'.format(plt_idx, digit_idx))
                    # plot progres
                    input_img = tf.reshape(self.xtr_record[:n_stim, digit_idx], (sqrt_nstim, sqrt_nstim)) / pamp
                    reconst_img = tf.reshape(
                        self.w['pc1'] @
                        tf.reshape(self.xtr_record[n_stim * 3:n_stim * 3 + self.n_pred[0], digit_idx],
                                   (self.xtr_record[n_stim * 3:n_stim * 3 + self.n_pred[0], digit_idx].shape[0], 1)),
                        (sqrt_nstim, sqrt_nstim)) / pamp

                    input_plot = axs[plt_idx, 0].imshow(input_img, cmap='Reds', vmin=1000, vmax=4000)
                    fig.colorbar(input_plot, ax=axs[plt_idx, 0], shrink=0.6)
                    reconst_plot = axs[plt_idx, 1].imshow(reconst_img, cmap='Reds', vmin=1000, vmax=4000)
                    fig.colorbar(reconst_plot, ax=axs[plt_idx, 1], shrink=0.6)
                    diff_plot = axs[plt_idx, 2].imshow(input_img - reconst_img, cmap='bwr',
                                                      vmin=-1000, vmax=1000)
                    fig.colorbar(diff_plot, ax=axs[plt_idx, 2], shrink=0.6)
                    plt_idx += 1
                    fig.suptitle('progress update: epoch #{0}/{1}'.format(epoch_i + 1, num_epoch))

            plt.show()

            # time remaining
            epoch_time_avg += time.time() - epoch_time
            print('***** time remaining = {0}'.format(
                str(timedelta(seconds=epoch_time_avg / (epoch_i + 1) * (num_epoch - epoch_i - 1)))))

            # calculate sse
            input_image = tf.reshape(self.xtr_record[:n_stim, :], (sqrt_nstim, sqrt_nstim, self.batch_size)) / pamp
            reconstructed_image = tf.reshape(
                self.w['pc1'] @ self.xtr_record[n_stim * 3:n_stim * 3 + self.n_pred[0], :],
                (sqrt_nstim, sqrt_nstim, self.batch_size)) / pamp
            sse.append(tf.reduce_sum(tf.reduce_mean((reconstructed_image - input_image) ** 2, axis=2)).numpy())

            sse_fig = plt.figure()
            plt.plot(np.log(sse))
            plt.xlabel('epoch #')
            plt.ylabel('log (SSE)')

        end_time = time.time()
        print('simulation : {0:.2f} sec'.format(end_time - start_time))

        return sse, sse_fig


pamp = 10 ** -12

# network parameters
n_pc_layers = 1
n_pred_neurons = [400, 225]

# create external input
batch_size = 10
n_shape = 3
n_samples = 20

ext_current, digits, test_set_idx, label_set_shuffled = create_mnist_set(nDigit=n_shape, nSample=n_samples, shuffle=True)
ext_current *= pamp
n_stim = ext_current.shape[1]
sqrt_nstim = int(np.sqrt(n_stim))

# plot the same test set
# plot_imgset(img_set, 10, n_shape, n_batch)
plot_mnist_set(testset=ext_current, testset_idx=test_set_idx, nDigit=n_shape, nSample=n_samples)
plt.show()

# # ext_current = img_set.reshape(n_shape * n_batch, n_stim).T

# build network
adex_01 = AdEx_Layer(neuron_model_constants=AdEx,
                     num_pc_layers=n_pc_layers,
                     num_pred_neurons=n_pred_neurons,
                     num_stim=n_stim,
                     num_gist=100)

# simulate
sim_dur = 500 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-4)  # ms
learning_window = 200 * 10 ** -3

n_epoch = 10
lrate = 0.25e-8
reg_alpha = 5e-3

# train_network(self, num_epoch, sim_dur, sim_dt, sim_lt, lr, reg_a, input_current, n_shape, n_batch, set_idx):
sse, sse_fig = adex_01.train_network(num_epoch=n_epoch, sim_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                                     lr=lrate, reg_a=reg_alpha,
                                     input_current=ext_current.T,
                                     n_shape=n_shape, batch_size=batch_size,
                                     set_idx=test_set_idx)
plt.show()
# print weight dist
w_fig = weight_dist(weights=adex_01.w['pc1'], weights_init=adex_01.w_init['pc1'])
plt.show()
# test_inference(self, imgs, stim_shape, stim_type, digit_list=None)
test_fig = adex_01.test_inference(imgs=ext_current.T,
                                  stim_shape=n_shape,
                                  stim_type='novel',
                                  sim_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                                  digit_list=digits)
plt.show()
