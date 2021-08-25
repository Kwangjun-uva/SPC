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
import glob
import re
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

def latest_epoch(file_dir):
    num = []
    for file in glob.glob(file_dir + '/progress_update*'):
        num.append(int(re.findall('[0-9]+', file)[-1]))
    return max(num)

def update_sim_time(folder, print_line):
    sim_time_txt = open(folder + '/sim_time.txt', 'a')
    sim_time_txt.write(print_line)


class AdEx_Layer(object):

    def __init__(self,
                 sim_directory,
                 neuron_model_constants,
                 num_pc_layers, num_pred_neurons,
                 num_stim,
                 gist_num,
                 w_mat):
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
        self.w = w_mat
        self.w_init = w_mat
        # connect
        # self.connect_pc()
        # self.connect_gist(conn_p=gist_connp, max_w=gist_maxw)

        # constant weight
        self.w_const = 550 * 10 ** -12
        # weight update time interval
        self.l_time = None


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
        self.fired = tf.Variable(tf.zeros([self.n_variable, self.batch_size], dtype=tf.float32))

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

        for t in range(int(self.T / self.dt)):
            # update internal variables (v, c, x, x_tr)
            self.update_var()
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
        self.fs.assign_add(self.fired)

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

            # index of current prediction layer
            curr_p_idx = sum(self.neurons_per_group[:pc_layer_idx * 3])
            curr_p_size = self.neurons_per_group[pc_layer_idx * 3]

            # index of
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

    # def hebbian_dw(self, source, target, lr, reg_alpha):
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
                      I_ext=data_set.T,
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

        # starting idx for progress update
        prg_start_idx = latest_epoch(save_folder)

        # number of batches
        n_batch = int(input_current.shape[1] / batch_size)

        start_time = time.time()
        # load sse from previous training
        with open(self.model_dir + '/sse_dict.pickle', 'rb') as sse_handle:
            sse_original = pickle.load(sse_handle)
        sse = {}
        for key, sse_pci in sse_original.items():
            sse[key] = sse_pci

        # # create a sse dictionary for pc layers
        # for i in range(1, self.n_pc_layer + 1):
        #     sse['pc' + str(i)] = []

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
                                'iter #{3}/{4} = {5:.2f} sec'.format(epoch_i + 1 + prg_start_idx,
                                                                     num_epoch + prg_start_idx,
                                                                     end_iter_time - epoch_time,
                                                                     iter_i + 1, n_batch,
                                                                     end_iter_time - iter_time))

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

                fig, axs = plt.subplots(ncols=self.n_pc_layer + 1, nrows=n_class, figsize=(4 * 5, 4 * n_class))
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
                fig.suptitle('progress update: epoch #{0}/{1}'.format(epoch_i + 1 + prg_start_idx,
                                                                      num_epoch + prg_start_idx))
                fig.savefig(self.model_dir + '/progress_update_{0:0=2d}.png'.format(epoch_i + 1 + prg_start_idx))
                plt.close(fig)

            # time remaining
            epoch_time_avg += time.time() - epoch_time
            update_sim_time(self.model_dir, '\n***** time remaining = {0}'.format(
                str(timedelta(seconds=epoch_time_avg / (epoch_i + 1) * (num_epoch - epoch_i - 1)))))

            sse_fig, sse_axs = plt.subplots(nrows=3, ncols=1, sharex=True)
            for i in range(1, self.n_pc_layer + 1):
                bu_start_idx = sum(self.neurons_per_group[:3 * (i - 1)])
                bu_end_idx = bu_start_idx + self.neurons_per_group[3*(i-1)]
                td_start_idx = sum(self.neurons_per_group[:3 * i])
                td_end_idx = td_start_idx + self.n_pred[i - 1]

                bu_input = self.xtr_record[bu_start_idx:bu_end_idx] / pamp
                td_pred = (self.w['pc' + str(i)] @ self.xtr_record[td_start_idx:td_end_idx]) / pamp

                sse['pc' + str(i)].append(tf.reduce_sum(tf.reduce_mean((td_pred - bu_input) ** 2, axis=1)).numpy())
                sse_axs[i - 1].plot(np.arange(epoch_i + 1 + prg_start_idx), np.log(sse['pc' + str(i)]))
                sse_axs[i - 1].set_xlabel('epoch #')
                sse_axs[i - 1].set_ylabel('log (SSE)')
                sse_axs[i - 1].label_outer()

            sse_fig.suptitle('SSE update: epoch #{0}/{1}'.format(epoch_i + 1 + prg_start_idx,
                                                                 num_epoch + prg_start_idx))
            sse_fig.savefig(self.model_dir + '/log_sse.png')#.format(epoch_i + 1 + prg_start_idx))
            plt.close(sse_fig)

            if (epoch_i+1) % n_plot_idx == 0:
                # weight dist change
                w_fig = weight_dist(savefolder=save_folder,
                                    weights=self.w, weights_init=self.w_init,
                                    n_pc=self.n_pc_layer)

                test_fig = self.test_inference(imgs=test_set,
                                                  nsample=test_n_sample, ndigit=n_class,
                                                  simul_dur=simul_dur, sim_dt=sim_dt, sim_lt=sim_lt,
                                                  train_or_test='test')

                # rdm analysis
                rdm_fig = rdm_plots(model=adex_01,
                                    testing_current=test_set, n_class=n_class,
                                    savefolder=save_folder, trained="test")

        end_time = time.time()
        update_sim_time(self.model_dir, '\nsimulation : {0}'.format(str(timedelta(seconds=end_time - start_time))))

        return sse


def pick_idx(idx_set, digits, size_batch):

    digit_cols = np.zeros(len(digits))
    rrr = []

    ll = idx_set[-size_batch:]
    for i, digit_i in enumerate(digits):
        ssi = ll.index(digit_i)
        if digit_cols[i] < 100:
            rrr.append(ssi)
            digit_cols[i] = 100

    return rrr


def conn_probs(n_a, n_b):
    return np.sqrt(n_b / n_a) * 0.025

def save_data(sim_name):

    # gather new learned weights
    save_ws = {}
    for key, ws in adex_01.w.items():
        save_ws[key] = ws.numpy()
    # replace learned weight dictionary
    with open(sim_name + '/weight_dict.pickle', 'wb') as w_handle:
        pickle.dump(save_ws, w_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # update SSE by appending
    # with open(sim_name + '/sse_dict.pickle', 'rb') as sse_handle:
    #     sse_original = pickle.load(sse_handle)
    # for key, sse_pci in sse_original.items():
    #     sse_original[key] = sse_pci + sse[key]
    # save
    with open(sim_name + '/sse_dict.pickle', 'wb') as sse_handle2:
        pickle.dump(sse, sse_handle2, protocol=pickle.HIGHEST_PROTOCOL)

    # update epoch length
    with open(sim_name + '/sim_params_dict.pickle', 'rb') as params_handle:
        params_original = pickle.load(params_handle)
    params_original['n_epoch'] += n_epoch
    # save
    with open(sim_name + '/sim_params_dict.pickle', 'wb') as handle2:
        pickle.dump(params_original, handle2, protocol=pickle.HIGHEST_PROTOCOL)

def matrix_rdm(matrix_data):
    output = np.array([1 - spearmanr(matrix_data[ni], matrix_data[mi])[0]
                       for ni in range(len(matrix_data))
                       for mi in range(len(matrix_data))]).reshape(len(matrix_data), len(matrix_data))

    return output


def rdm_plots(model, testing_current, n_class, savefolder, trained):
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

    rdm1_filename = 'rdm1_dict'
    rdm2_filename = 'rdm2_dict'
    if os.path.isfile(savefolder + '/' + rdm1_filename + '.pickle'):
        rdm1_filename += datetime.today().strftime('_%Y_%m_%d_%H_%M')
    with open(savefolder + '/' + rdm1_filename + '.pickle', 'wb') as handle:
        pickle.dump(rdms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if os.path.isfile(savefolder + '/' + rdm2_filename + '.pickle'):
        rdm2_filename += datetime.today().strftime('_%Y_%m_%d_%H_%M')
    with open(savefolder + '/' + rdm2_filename + '.pickle', 'wb') as handle:
        pickle.dump(rdms, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    rdm_fig_name = 'rdms_plot_' + trained
    if os.path.isfile(savefolder + '/' + rdm_fig_name + '.pickle'):
        rdm_fig_name += datetime.today().strftime('_%Y_%m_%d_%H_%M')
    fig.savefig(savefolder + '/' + rdm_fig_name + '.png')

    return fig


# load constants
with open('adex_constants.pickle', 'rb') as f:
    AdEx = pickle.load(f)

# specify the folder
save_folder = sys.argv[1]

# load training dictionary
training_dict = {}
for i,j in list(np.load(save_folder + '/training_dict.npz').items()):
    training_dict[i] = j
locals().update(training_dict)

# load simulation and network parameters
with open(save_folder + '/sim_params_dict.pickle', 'rb') as sim_pm:
    sim_params = pickle.load(sim_pm)
locals().update(sim_params)

# load learned weights from previous training
with open(save_folder + '/weight_dict.pickle', 'rb') as wdict:
    w_mat = pickle.load(wdict)
# convert them to tensors
for key, grp in w_mat.items():
    w_mat[key] = tf.convert_to_tensor(grp)


# unit
pamp = 10 ** -12

# not necessary from next training
n_epoch = int(sys.argv[2])

# training_set, training_labels, test_set, test_labels, digits, training_set_idx
training_set = np.load(save_folder + '/training_data.npy')
test_set = np.load(save_folder + '/test_data.npy')

n_stim = training_set.shape[1]
sqrt_nstim = int(np.sqrt(n_stim))

conn_vals = np.array([conn_probs(a_i, b_i)
                              for a_i, b_i in zip([n_stim] + [n_gist] * n_pc_layers, [n_gist] + n_pred_neurons)]) * 0.05

max_vals = np.array([1] * (n_pc_layers + 1)) * 0.25 * 5

# test inference on test data
test_n_shape = n_shape
test_n_sample = 16
test_iter_idx = int(n_samples/test_n_sample)

testing_set = test_set[::test_iter_idx]

# # create a folder to save results
# save_folder = 'sparseness'
# if os.path.exists(save_folder):
#     save_folder += datetime.today().strftime('_%Y_%m_%d_%H_%M')
# os.mkdir(save_folder)

gpus = tf.config.experimental.list_logical_devices('GPU')
gpu_i = int(sys.argv[3])
# gpu_i = 0

with tf.device(gpus[gpu_i].name):

    # build network
    adex_01 = AdEx_Layer(sim_directory=save_folder,
                         neuron_model_constants=AdEx,
                         num_pc_layers=n_pc_layers,
                         num_pred_neurons=n_pred_neurons,
                         num_stim=n_stim,
                         gist_num=n_gist, w_mat=w_mat)

    # train_network(self, num_epoch, sim_dur, sim_dt, sim_lt, lr, reg_a, input_current, n_shape, n_batch, set_idx):
    sse = adex_01.train_network(num_epoch=n_epoch,
                                simul_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                                lr=lrate, reg_a=reg_alpha,
                                input_current=training_set.T,
                                test_set=testing_set, test_n_sample=test_n_sample,
                                n_class=n_shape, batch_size=batch_size,
                                set_idx=rep_set_idx, report_idx=report_index, n_plot_idx=n_plot_idx)

    # save simulation data
    save_data(save_folder)


# # weight dist change : !!!! need to fix the initial weights. needs to be loaded from a separate init weight dict !!!!
# with open(save_folder + '/weight_init_dict.pickle', 'rb') as wdict_init:
#     w_init_mat = pickle.load(wdict_init)
# # convert them to tensors
# for key, grp in w_init_mat.items():
#     w_init_mat[key] = tf.convert_to_tensor(grp)

# w_fig = weight_dist(savefolder=save_folder,
#                     weights=adex_01.w, weights_init=w_init_mat,
#                     n_pc=adex_01.n_pc_layer)
#
# test_fig = adex_01.test_inference(imgs=testing_set,
#                                   nsample=test_n_sample, ndigit=test_n_shape,
#                                   simul_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
#                                   train_or_test='test')
# # rdm analysis
# rdm_fig = rdm_plots(model=adex_01,
#                     testing_current=testing_set, n_class=test_n_shape,
#                     savefolder=save_folder, trained="test")