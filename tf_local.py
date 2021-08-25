import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from test_func import weight_dist
from mnist_data import create_mnist_set, plot_mnist_set
from datetime import timedelta
import pickle5 as pickle
from scipy.stats import spearmanr
import os
import sys
from datetime import datetime


# A basic adex LIF neuron
def plot_pc1rep(input_img, l1rep, nDigit, nSample):
    testset_x, testset_y, testset_len = l1rep.shape

    l1_img = np.zeros(((testset_x + 2) * nSample, (testset_y + 2) * nDigit))
    inp_img = np.zeros(((testset_x + 2) * nSample, (testset_y + 2) * nDigit))
    for i in range(nSample):
        ix = (testset_x + 2) * i + 1
        for j in range(nDigit):
            iy = (testset_y + 2) * j + 1
            l1_img[ix:ix + testset_x, iy:iy + testset_y] = l1rep[:, :, i + j * nSample]
            inp_img[ix:ix + testset_x, iy:iy + testset_y] = input_img[:, :, i + j * nSample]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(nDigit * 1, nSample * 1))
    axs[0].imshow(inp_img, cmap='Reds', vmin=600, vmax=3000)
    axs[0].axis('off')
    axs[0].set_title('Input MNIST images')
    axs[1].imshow(l1_img, cmap='Reds', vmin=600, vmax=3000)
    axs[1].axis('off')
    axs[1].set_title('L1 representations of MNIST images')
    fig.tight_layout()

    return fig


def update_sim_time(folder, print_line):
    sim_time_txt = open(folder + '/sim_time.txt', 'a')
    sim_time_txt.write(print_line)


class AdEx_Layer(object):

    def __init__(self,
                 sim_directory,
                 neuron_model_constants,
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

        self.model_dir = sim_directory

        for key in neuron_model_constants:
            setattr(self, key, neuron_model_constants[key])

        # network architecture
        self.n_pc_layer = num_pc_layers
        self.n_pred = num_pred_neurons
        self.n_gist = gist_num
        self.n_stim = num_stim

        # self.n_groups = num_pc_layers * 3 + 1
        self.neurons_per_group = [self.n_stim] * 3 + np.repeat([self.n_pred[:-1]], 3).tolist() + [self.n_pred[-1]] + [
            self.n_gist]
        self.n_variable = sum(self.neurons_per_group)

        # initial weight preparation
        self.w = {}
        self.w_init = {}
        # connect
        self.connect_pc()
        self.connect_gist(conn_p=gist_connp, max_w=gist_maxw)

        # constant weight
        self.w_const = 550 * 10 ** -12
        # weight update time interval
        self.l_time = None

        # # internal variables
        # self.v = None
        # self.c = None
        # self.ref = None
        # # pre-synaptic variables
        # self.x = None
        # self.x_tr = None
        # # post-synaptic variable
        # self.Isyn = None
        # self.fired = None
        # self.xtr_record = None

    def initialize_var(self):

        # internal variables
        self.v = tf.Variable(tf.ones([self.n_variable, self.batch_size], dtype=tf.float32) * self.EL)
        self.c = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.ref = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        # pre-synaptic variables
        self.x = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.x_tr = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        # post-synaptic variable
        self.Isyn = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        self.fired = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.bool))

        # self.xtr_record = None

    def __call__(self, sim_duration, time_step, lt, I_ext, bat_size):

        # simulation parameters
        self.T = sim_duration
        self.dt = time_step

        self._step = 0
        self.l_time = lt

        self.batch_size = bat_size

        # initialize internal variables
        self.initialize_var()

        # feed external corrent to the first layer
        self.Iext = tf.constant(I_ext, dtype=tf.float32)
        # self.fs = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))
        # self.xtr_s = tf.Variable(tf.zeros([int(self.T / self.dt)], dtype=tf.float32))
        # self.xxx = tf.Variable(tf.zeros([int(self.T / self.dt)], dtype=tf.float32))

        for t in range(int(self.T / self.dt)):
            # update internal variables (v, c, x, x_tr)
            self.update_var()
            # self.xtr_s[t].assign(self.x_tr[784*3+100, 1])
            # update synaptic variable (Isyn = w * x_tr + Iext)
            self.record_pre_post()

            self._step += 1

        # take the mean of synaptic output
        self.xtr_record.assign(self.xtr_record / (self.l_time / self.dt))

    def update_var(self):

        # feed synaptic current to higher layers
        self.update_Isyn()

        # current refractory status [0,2] ms
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
        self.fired = tf.cast(tf.greater_equal(self.v, self.VT), tf.float32)
        # reset variables
        self.v = self.fired * self.EL + (1 - self.fired) * self.v
        self.c = self.fired * tf.add(self.c, self.b) + (1 - self.fired) * self.c
        self.x = self.fired * -self.x_reset + (1 - self.fired) * self.x

        # set lower boundary of v (Vrest = -70.6 mV)
        self.v = tf.maximum(self.EL, self.v)
        self.ref = tf.add(self.ref, self.fired * float(self.t_ref / self.dt))

    def update_v(self, constraint):
        dv = (self.dt / self.Cm) * (self.gL * (self.EL - self.v) +
                                    self.gL * self.DeltaT * tf.exp((self.v - self.VT) / self.DeltaT) +
                                    self.Isyn - self.c)
        dv_ref = (1 - constraint) * dv
        self.v = tf.add(self.v, dv_ref)

    def update_c(self, constraint):
        dc = (self.dt / self.tauw) * (self.a * (self.v - self.EL) - self.c)
        dc_ref = (1 - constraint) * dc
        self.c = tf.add(self.c, dc_ref)

    def update_x(self):
        dx = self.dt * (-self.x / self.tau_rise)
        self.x = tf.add(self.x, dx)

    def update_xtr(self):
        dxtr = self.dt * (-self.x / self.tau_rise - self.x_tr / self.tau_s)
        self.x_tr = tf.add(self.x_tr, dxtr)

    def update_Isyn(self):

        # I = ext
        self.Isyn[:self.n_stim].assign(self.Iext)
        # gist = W[ig]@ Isyn[I]
        if self._step < 500:
            input_gist = tf.transpose(self.w['ig']) @ (self.x_tr[:self.neurons_per_group[0]] * self.w_const)
            self.Isyn[-self.n_gist:, :].assign(input_gist)
        else:
            self.Isyn[-self.n_gist:, :].assign(tf.zeros(shape=self.Isyn[-self.n_gist:, :].shape, dtype=tf.float32))

        for pc_layer_idx in range(self.n_pc_layer):
            self.Isyn_by_layer(pc_layer_idx)

    def Isyn_by_layer(self, pc_layer_idx):
        # index of current prediction layer
        curr_p_idx = sum(self.neurons_per_group[:pc_layer_idx * 3])
        curr_p_size = self.neurons_per_group[pc_layer_idx * 3]

        # index of next prediction layer
        next_p_idx = sum(self.neurons_per_group[:pc_layer_idx * 3 + 3])
        next_p_size = self.neurons_per_group[pc_layer_idx * 3 + 3]

        # input / predictin error
        bu_sensory = self.x_tr[curr_p_idx: curr_p_idx + curr_p_size, :] * self.w_const
        # prediction
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
        gist = tf.transpose(self.w['gp' + str(pc_layer_idx + 1)]) @ (self.x_tr[-self.n_gist:, :] * self.w_const)

        if pc_layer_idx < self.n_pc_layer - 1:
            td_err_pos = self.x_tr[next_p_idx + next_p_size:next_p_idx + 2 * next_p_size] * self.w_const
            td_err_neg = self.x_tr[next_p_idx + 2 * next_p_size:next_p_idx + 3 * next_p_size] * self.w_const
            self.Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(
                tf.add(
                    tf.add(
                        tf.add(bu_err_pos, -bu_err_neg),
                        tf.add(-td_err_pos, td_err_neg)),
                    gist))
        else:
            self.Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(tf.add(tf.add(bu_err_pos, -bu_err_neg), gist))

    def connect_pc(self):

        for pc_layer_idx in range(self.n_pc_layer):
            err_size = self.neurons_per_group[pc_layer_idx * 3 + 1]
            pred_size = self.n_pred[pc_layer_idx]

            norm_factor = 0.1 * pred_size

            # self.w['pc' + str(pc_layer_idx + 1)] = tf.random.normal((err_size, pred_size), 1.0, 0.3) / norm_factor
            self.w['pc' + str(pc_layer_idx + 1)] = tf.abs(tf.random.normal((err_size, pred_size), 0.0, 0.3) / norm_factor)
            self.w_init['pc' + str(pc_layer_idx + 1)] = self.w['pc' + str(pc_layer_idx + 1)]

    def connect_gist(self, conn_p, max_w):
        """
        :param conn_p: list of float. connection probabilities ranges between [0,1]
        :param max_w: list of int. max weight values.
        :return: w['ig'] and w['gp']
        """
        # needs to be shortened!!!
        ig_shape = (self.neurons_per_group[0], self.n_gist)
        rand_w = tf.random.normal(shape=ig_shape, mean=max_w[0], stddev=1 / self.n_stim, dtype=tf.float32)
        constraint = tf.cast(tf.greater(tf.random.uniform(shape=ig_shape), 1 - conn_p[0]), tf.float32)
        self.w['ig'] = constraint * rand_w

        for pc_layer_idx in range(self.n_pc_layer):
            gp_shape = (self.n_gist, self.n_pred[pc_layer_idx])
            rand_w = tf.random.normal(shape=gp_shape, mean=max_w[pc_layer_idx + 1],
                                      stddev=1 / self.n_pred[pc_layer_idx], dtype=tf.float32)
            constraint = tf.cast(tf.greater(tf.random.uniform(shape=gp_shape), 1 - conn_p[pc_layer_idx + 1]),
                                 tf.float32)
            self.w['gp' + str(pc_layer_idx + 1)] = constraint * rand_w

    def record_pre_post(self):

        if self._step == int(self.T / self.dt) - int(self.l_time / self.dt):

            self.xtr_record = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))

        elif self._step > int(self.T / self.dt) - int(self.l_time / self.dt):

            self.xtr_record.assign_add(self.x_tr * self.w_const)

    def weight_update(self, lr, alpha_w):

        for pc_layer_idx in range(self.n_pc_layer):

            err_idx = sum(self.neurons_per_group[:pc_layer_idx * 3 + 1])
            err_size = self.neurons_per_group[pc_layer_idx * 3 + 1]
            pred_idx = sum(self.neurons_per_group[:pc_layer_idx * 3 + 3])
            pred_size = self.n_pred[pc_layer_idx]

            xtr_ep = self.xtr_record[err_idx: err_idx + err_size]
            xtr_en = self.xtr_record[err_idx + err_size: err_idx + 2 * err_size]
            xtr_p = self.xtr_record[pred_idx: pred_idx + pred_size]

            dw_all_pos = lr[pc_layer_idx] * tf.einsum('ij,kj->ikj', xtr_ep / 10 ** -12, xtr_p / 10 ** -12)
            dw_all_neg = lr[pc_layer_idx] * tf.einsum('ij,kj->ikj', xtr_en / 10 ** -12, xtr_p / 10 ** -12)

            dw_mean_pos = tf.reduce_mean(dw_all_pos, axis=2) - 2 * alpha_w[pc_layer_idx] * tf.abs(
                self.w['pc' + str(pc_layer_idx + 1)])
            dw_mean_neg = tf.reduce_mean(dw_all_neg, axis=2) - 2 * alpha_w[pc_layer_idx] * tf.abs(
                self.w['pc' + str(pc_layer_idx + 1)])

            dws = tf.add(dw_mean_pos, -dw_mean_neg)

            self.w['pc' + str(pc_layer_idx + 1)] = tf.maximum(tf.add(self.w['pc' + str(pc_layer_idx + 1)], dws), 0.0)

    def test_inference(self, imgs, ndigit, nsample,
                       simul_dur, sim_dt, sim_lt, train_or_test):

        data_set = imgs

        # load the model
        self.__call__(sim_duration=simul_dur, time_step=sim_dt, lt=sim_lt,
                      I_ext=data_set.T * 10 ** -12,
                      bat_size=data_set.shape[0])

        input_shape = int(np.sqrt(self.n_stim))
        input_image = tf.reshape(self.xtr_record[:self.n_stim, :],
                                 (input_shape, input_shape, data_set.shape[0])) / pamp
        reconstructed_image = tf.reshape(
            self.w['pc1'] @ self.xtr_record[self.n_stim * 3:self.n_stim * 3 + self.n_pred[0], :],
            (input_shape, input_shape, data_set.shape[0])) / pamp

        l1rep_fig = plot_pc1rep(input_image.numpy(), reconstructed_image.numpy(), ndigit, nsample)
        l1rep_fig.savefig(self.model_dir + '/{0}_nD{1}_nS{2}.png'.format(train_or_test, ndigit, nsample))

        return l1rep_fig

    def train_network(self,
                      num_epoch, simul_dur, sim_dt, sim_lt,
                      lr, reg_a,
                      input_current,
                      test_set, test_n_sample,
                      n_class, batch_size,
                      set_idx,
                      report_idx, n_plot_idx):

        # number of batches
        n_batch = int(input_current.shape[1] / batch_size)

        start_time = time.time()
        # create a sse dictionary for pc layers
        self.sse = {}
        for i in range(1, self.n_pc_layer + 1):
            self.sse['pc' + str(i)] = []

        # initialize epoch sim time average
        epoch_time_avg = 0

        for epoch_i in range(num_epoch):

            epoch_time = time.time()

            for iter_i in range(n_batch):
                iter_time = time.time()
                curr_batch = input_current[:, iter_i * batch_size:(iter_i + 1) * batch_size]

                self.__call__(sim_duration=simul_dur, time_step=sim_dt, lt=sim_lt,
                              I_ext=curr_batch,
                              bat_size=batch_size)

                # update weightss
                self.weight_update(lr=lr, alpha_w=reg_a)

                end_iter_time = time.time()

                update_sim_time(self.model_dir,
                                '\nepoch #{0}/{1} = {2:.2f}, '
                                'iter #{3}/{4} = {5:.2f} sec'.format(epoch_i + 1, num_epoch,
                                                                     end_iter_time - epoch_time,
                                                                     iter_i + 1, n_batch, end_iter_time - iter_time))

            if ((epoch_i + 1) % report_idx == 0):  # and (len(set_idx) > iter_i):
                set_id = set_idx  # [iter_i]
                # plot progres
                input_img = self.xtr_record[:self.n_stim].numpy()[:, set_id].reshape(sqrt_nstim, sqrt_nstim,
                                                                                     len(set_id)) / pamp
                reconst_img = (self.w['pc1'].numpy() @
                               self.xtr_record[n_stim * 3:
                                               n_stim * 3 + self.n_pred[0]].numpy()[:, set_id]).reshape(sqrt_nstim,
                                                                                                        sqrt_nstim,
                                                                                                        len(set_id)) / pamp

                fig, axs = plt.subplots(ncols=3, nrows=n_class, figsize=(4 * 3, 4 * n_class))
                for plt_idx in range(len(set_id)):
                    input_plot = axs[plt_idx, 0].imshow(input_img[:, :, plt_idx], cmap='Reds', vmin=600, vmax=4000)
                    fig.colorbar(input_plot, ax=axs[plt_idx, 0], shrink=0.6)
                    reconst_plot = axs[plt_idx, 1].imshow(reconst_img[:, :, plt_idx], cmap='Reds', vmin=600, vmax=4000)
                    fig.colorbar(reconst_plot, ax=axs[plt_idx, 1], shrink=0.6)
                    diff_plot = axs[plt_idx, 2].imshow(input_img[:, :, plt_idx] - reconst_img[:, :, plt_idx],
                                                       cmap='bwr',
                                                       vmin=-1000, vmax=1000)
                    fig.colorbar(diff_plot, ax=axs[plt_idx, 2], shrink=0.6)

                [axi.axis('off') for axi in axs.ravel()]
                fig.suptitle('progress update: epoch #{0}/{1}'.format(epoch_i + 1, num_epoch))
                fig.savefig(self.model_dir + '/progress_update_{0:0=2d}.png'.format(epoch_i + 1))
                plt.close(fig)

            # time remaining
            epoch_time_avg += time.time() - epoch_time
            update_sim_time(self.model_dir, '\n***** time remaining = {0}'.format(
                str(timedelta(seconds=epoch_time_avg / (epoch_i + 1) * (num_epoch - epoch_i - 1)))))
            # print('***** time remaining = {0}'.format(
            #     str(timedelta(seconds=epoch_time_avg / (epoch_i + 1) * (num_epoch - epoch_i - 1)))))

            # calculate sse
            # l0_input = tf.reshape(self.xtr_record[:n_stim, :], (sqrt_nstim, sqrt_nstim, self.batch_size)) / pamp
            # l1_pred = tf.reshape(
            #     self.w['pc1'] @ self.xtr_record[n_stim * 3:n_stim * 3 + self.n_pred[0], :],
            #     (sqrt_nstim, sqrt_nstim, self.batch_size)) / pamp

            sse_fig, sse_axs = plt.subplots(nrows=self.n_pc_layer, ncols=1, sharex=True)
            for i in range(1, self.n_pc_layer + 1):
                bu_start_idx = sum(self.neurons_per_group[:3 * (i - 1)])
                bu_end_idx = bu_start_idx + self.neurons_per_group[3*(i-1)]
                td_start_idx = sum(self.neurons_per_group[:3 * i])
                td_end_idx = td_start_idx + self.n_pred[i - 1]

                bu_input = self.xtr_record[bu_start_idx:bu_end_idx] / pamp
                td_pred = (self.w['pc' + str(i)] @ self.xtr_record[td_start_idx:td_end_idx]) / pamp

                self.sse['pc' + str(i)].append(tf.reduce_sum(tf.reduce_mean((td_pred - bu_input) ** 2, axis=1)).numpy())
                sse_axs[i - 1].plot(np.arange(epoch_i + 1), np.log(self.sse['pc' + str(i)]))
                sse_axs[i - 1].set_xlabel('epoch #')
                sse_axs[i - 1].set_ylabel('log (SSE)')
                sse_axs[i - 1].label_outer()

            sse_fig.suptitle('SSE update: epoch #{0}/{1}'.format(epoch_i + 1, num_epoch))
            sse_fig.savefig(self.model_dir + '/log_sse.png')
            plt.close(sse_fig)

            if (epoch_i+1) % n_plot_idx == 0:
                # weight dist change
                w_fig = weight_dist(savefolder=save_folder,
                                    weights=self.w, weights_init=self.w_init,
                                    n_pc=self.n_pc_layer, epoch_i=epoch_i)

                test_fig = self.test_inference(imgs=test_set,
                                              nsample=test_n_sample, ndigit=n_class,
                                              simul_dur=simul_dur, sim_dt=sim_dt, sim_lt=sim_lt,
                                              train_or_test='test')

                # rdm analysis
                rdm_fig = rdm_plots(model=adex_01,
                                    testing_current=test_set, n_class=n_class,
                                    savefolder=save_folder, trained="test",
                                    epoch_i=epoch_i)

                # save simulation data
                save_data(save_folder)

        end_time = time.time()
        update_sim_time(self.model_dir, '\nsimulation : {0}'.format(str(timedelta(seconds=end_time - start_time))))

        return self.sse


def pick_idx(idx_set, digits, size_batch):
    # return_list = []

    n_batch = int(len(idx_set) / size_batch)

    # ll = [idx_set[i * size_batch:i * size_batch + size_batch] for i in range(n_batch)]
    digit_cols = np.zeros(len(digits))
    rrr = []
    #
    # # # loop over batches (i.e. number of batches = len(ll))
    # # for j in range(len(ll)):
    # #     rri = []
    # #     # loop over classes
    # #     for i, digit_i in enumerate(digits):
    # #         if digit_i in ll[j]:
    # #             ssi = ll[j].index(digit_i)
    # #             if digit_cols[i] < 100:
    # #                 rri.append(ssi)
    # #                 digit_cols[i] = 100
    # #     if rri:
    # #         rrr.append(rri)
    ll = idx_set[-size_batch:]
    for i, digit_i in enumerate(digits):
        ssi = ll.index(digit_i)
        if digit_cols[i] < 100:
            rrr.append(ssi)
            digit_cols[i] = 100

    return rrr


def conn_probs(n_a, n_b):
    return np.sqrt(n_b / n_a) * 0.025


def save_results(sim_name):
    # save weights
    save_ws = {}
    for key, ws in adex_01.w.items():
        save_ws[key] = ws.numpy()
    with open(sim_name + '/weight_dict.pickle', 'wb') as w_handle:
        pickle.dump(save_ws, w_handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save initial weights
    save_ws_init = {}
    for key, ws in adex_01.w_init.items():
        save_ws_init[key] = ws.numpy()
    with open(sim_name + '/weight_init_dict.pickle', 'wb') as w_init_handle:
        pickle.dump(save_ws_init, w_init_handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save sse
    with open(sim_name + '/sse_dict.pickle', 'wb') as sse_handle:
        pickle.dump(adex_01.sse, sse_handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_data(sim_name):
    # save training data
    np.save(sim_name + '/training_data', training_set)
    np.save(sim_name + '/test_data', test_set)

    np.savez(sim_name + '/training_dict',
             digits=classes,
             training_set_idx=training_set_idx,
             training_labels=training_labels,
             test_set_labels=test_labels,
             rep_set_idx=rep_set_idx)
    # np.savez(sim_name + '/test_dict',
    #          digits=classes,
    #          test_set_idx=training_set_idx,
    #          training_labels=test_labels)

    # save simulation params
    sim_params = {'n_pc_layers': n_pc_layers, 'n_pred_neurons': n_pred_neurons, 'n_gist': n_gist,
                  'batch_size': batch_size, 'n_sample': n_samples, 'n_shape': n_shape, 'sim_dur': sim_dur, 'dt': dt,
                  'learning_window': learning_window, 'n_epoch': n_epoch, 'lrate': lrate, 'reg_alpha': reg_alpha,
                  'report_index': report_index}
    with open(sim_name + '/sim_params_dict.pickle', 'wb') as handle2:
        pickle.dump(sim_params, handle2, protocol=pickle.HIGHEST_PROTOCOL)


def matrix_rdm(matrix_data):
    output = np.array([1 - spearmanr(matrix_data[ni], matrix_data[mi])[0]
                       for ni in range(len(matrix_data))
                       for mi in range(len(matrix_data))]).reshape(len(matrix_data), len(matrix_data))

    return output


def rdm_plots(model, testing_current, n_class, savefolder, trained, epoch_i=None):

    inp_size, sample_size = testing_current.T.shape

    # RDM for input
    rdms = {'input': matrix_rdm(model.xtr_record[:model.n_stim, :].numpy().T)}

    # RDM for each PC layer
    for pc_i in range(1, model.n_pc_layer + 1):
        source_p_size = model.n_pred[pc_i - 1]

        curr_p_start_idx = sum(model.neurons_per_group[:3 * pc_i])
        curr_p_end_idx = sum(model.neurons_per_group[:3 * pc_i]) + source_p_size

        rdms['P' + str(pc_i)] = matrix_rdm((model.w['pc' + str(pc_i)] @
                                           model.xtr_record[curr_p_start_idx:curr_p_end_idx, :]).numpy().T)

    # RDM for an ideal classifier
    tmat = np.ones((sample_size, sample_size))
    nsample = int(sample_size / n_class)
    for i in range(n_class):
        tmat[i * nsample: i * nsample + nsample, i * nsample: i * nsample + nsample] = 0
    rdms['ideal_classifier'] = tmat

    # 2nd order RDMs
    rdms2 = [1 - spearmanr(rdms['input'].flatten(),
                           rdms['P' + str(pc_i)].flatten())[0]
             for pc_i in range(1, model.n_pc_layer + 1)] + \
            [1 - spearmanr(rdms['ideal_classifier'].flatten(),
                           rdms['P' + str(pc_i)].flatten())[0]
             for pc_i in range(1, model.n_pc_layer + 1)]

    with open(savefolder + '/rdm1_dict.pickle', 'wb') as handle:
        pickle.dump(rdms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(savefolder + '/rdm2_dict.pickle', 'wb') as handle:
        pickle.dump(rdms2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(nrows=2, ncols=2 * (model.n_pc_layer + 2), wspace=0.1)

    # plot rdms at each pc layer
    for col_idx, (keys, rdm_i) in enumerate(rdms.items()):
        rdm_plot = fig.add_subplot(gs[0, col_idx * 2: col_idx * 2 + 2])
        rdm_plot.imshow(rdm_i, cmap='Reds', aspect='auto', vmin=0, vmax=1)
        rdm_plot.set_title(keys)
        rdm_plot.set(xlabel='digit #', ylabel='digit#')
        rdm_plot.axis('off')
        rdm_plot.label_outer()

    # plot deviation from input
    r2_i = fig.add_subplot(gs[1, :model.n_pc_layer + 2])
    r2_i.bar(np.arange(model.n_pc_layer), rdms2[:model.n_pc_layer])
    r2_i.set_title('deviation from input')
    r2_i.set_xticks(np.arange(model.n_pc_layer))
    r2_i.set_xticklabels([pc_i for pc_i in rdms.keys() if 'P' in pc_i])
    r2_i.set_ylim([0, 1])
    r2_i.set_ylabel('1-Spearman corr')
    r2_i.label_outer()
    # plot deviation from ideal classifier
    r2_t = fig.add_subplot(gs[1, model.n_pc_layer + 2:])
    r2_t.bar(np.arange(model.n_pc_layer), rdms2[model.n_pc_layer:])
    r2_t.set_title('deviation from ideal classifier')
    r2_t.set_xticks(np.arange(model.n_pc_layer))
    r2_t.set_xticklabels([pc_i for pc_i in rdms.keys() if 'P' in pc_i])
    r2_t.set_ylim([0, 1])

    save_filename = '{}/rdms_plot_{}.png'.format(savefolder, trained)
    if epoch_i is not None:
        save_filename = '{}/rdms_plot_{}_{:03d}.png'.format(savefolder, trained, epoch_i + 1)

    fig.savefig(save_filename)

    return fig


# load constants
with open('adex_constants.pickle', 'rb') as f:
    AdEx = pickle.load(f)

# unit
pamp = 10 ** -12

# network parameters
n_pred_neurons = [900, 400, 225] # preferably each entry is an integer that has an integer square root
n_pc_layers = len(n_pred_neurons)
n_gist = 128

# create external input
batch_size = 256
n_shape = 3
n_samples = 256

# simulate
sim_dur = 500 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-4)  # ms
learning_window = 100 * 10 ** -3
report_index = 1

n_epoch = 3
lrate = np.repeat(1.0, 3) * 10 ** - 7
reg_alpha = np.repeat(1.0, 3) * 10 ** -3

keras_data = tf.keras.datasets.mnist
training_set, training_labels, test_set, test_labels, classes, training_set_idx = create_mnist_set(data_type=keras_data,
                                                                                                   nDigit=n_shape,
                                                                                                   nSample=n_samples,
                                                                                                   shuffle=True)
training_set *= pamp
n_stim = training_set.shape[1]
sqrt_nstim = int(np.sqrt(n_stim))

rep_set_idx = pick_idx(training_labels, classes, batch_size)
n_plot_idx = 1

conn_vals = np.array([conn_probs(a_i, b_i)
                              for a_i, b_i in zip([n_stim] + [n_gist] * n_pc_layers, [n_gist] + n_pred_neurons)]) * 0.05

max_vals = np.array([1] * (n_pc_layers + 1)) * 0.25 * 5

# test inference on test data
test_n_shape = n_shape
test_n_sample = 16
test_iter_idx = int(n_samples/test_n_sample)

testing_set = test_set[::test_iter_idx]

# create a folder to save results
save_folder = datetime.today().strftime('%Y_%m_%d_%H_%M') + '_nD' + str(n_shape) + 'nS' + str(n_samples) + 'nEP' + str(n_epoch)
os.mkdir(save_folder)

# plot the same test set
plot_mnist_set(testset=training_set, testset_idx=training_set_idx,
               nDigit=n_shape, nSample=n_samples,
               savefolder=save_folder)

# build network
adex_01 = AdEx_Layer(sim_directory=save_folder,
                     neuron_model_constants=AdEx,
                     num_pc_layers=n_pc_layers,
                     num_pred_neurons=n_pred_neurons,
                     num_stim=n_stim,
                     gist_num=n_gist, gist_connp=conn_vals, gist_maxw=max_vals)


# train_network(self, num_epoch, sim_dur, sim_dt, sim_lt, lr, reg_a, input_current, n_shape, n_batch, set_idx):
sse = adex_01.train_network(num_epoch=n_epoch,
                            simul_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                            lr=lrate, reg_a=reg_alpha,
                            input_current=training_set.T,
                            test_set=testing_set, test_n_sample=test_n_sample,
                            n_class=n_shape, batch_size=batch_size,
                            set_idx=rep_set_idx, report_idx=report_index,
                            n_plot_idx=n_plot_idx)

# # save simulation data
# save_data(save_folder)

# # weight dist change
# w_fig = weight_dist(savefolder=save_folder,
#                     weights=adex_01.w, weights_init=adex_01.w_init,
#                     n_pc=adex_01.n_pc_layer)
#
# test_fig = adex_01.test_inference(imgs=testing_set,
#                                   nsample=test_n_sample, ndigit=test_n_shape,
#                                   simul_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
#                                   train_or_test='test')
#
#
# # rdm analysis
# rdm_fig = rdm_plots(model=adex_01,
#                     testing_current=testing_set, n_class=test_n_shape,
#                     savefolder=save_folder, trained="test")
