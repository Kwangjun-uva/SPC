from AdEx_const import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tf_local import AdEx_Layer
from test_func import weight_dist
from datetime import timedelta
import pickle5 as pickle
import glob
import re
import sys

def latest_epoch(file_dir):
    num = []
    for file in glob.glob(file_dir + '/progress_update*'):
        num.append(int(re.findall('[0-9]+', file)[-1]))
    return max(num)

def update_sim_time(folder, print_line):
    sim_time_txt = open(folder + '/sim_time.txt', 'a')
    sim_time_txt.write(print_line)

def new_init(self, sim_directory,
             # neuron_model_constants,
             num_pc_layers, num_pred_neurons, num_stim, gist_num,
             w_mat, w_mat_init):

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
        self.w_init = w_mat_init

        # constant weight
        # weight update time interval
        self.l_time = None


def new_train(self,
                  num_epoch, simul_dur, sim_dt, sim_lt,
                  lr, reg_a,
                  input_current,
                  test_set, test_n_sample,
                  n_class, batch_size,
                  set_idx,
                  report_idx, n_plot_idx):

    plt.close('all')

    # starting idx for progress update
    prg_start_idx = latest_epoch(save_folder)

    # number of batches
    n_batch = int(input_current.shape[1] / batch_size)

    start_time = time.time()
    # load sse from previous training
    with open(self.model_dir + '/sse_dict.pickle', 'rb') as sse_handle:
        sse_original = pickle.load(sse_handle)
    self.sse = {}
    for key, sse_pci in sse_original.items():
        self.sse[key] = sse_pci

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

            # update weights
            if (epoch_i + 1) % 10 == 0:
                lr = [lr[pc_i] * (self.sse['pc' + str(pc_i + 1)][epoch_i - 1] / np.max(self.sse['pc' + str(pc_i + 1)]))
                      for pc_i in range(self.n_pc_layer)]
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
            neurons_per_pc = self.neurons_per_group[::3]
            fig, axs = plt.subplots(ncols=3 * self.n_pc_layer, nrows=n_class, figsize=(4 * 3 * self.n_pc_layer, 4 * n_class))
            for pc_i in range(self.n_pc_layer):  # for a 3-PC model, 0, 1, 2
                # plot progres : p1 - p3
                inp_size = int(np.sqrt(neurons_per_pc[pc_i]))
                input_img = self.xtr_record[sum(self.neurons_per_group[:3 * pc_i]):
                                            sum(self.neurons_per_group[:3 * pc_i])
                                            + neurons_per_pc[pc_i]].numpy()[:, set_id].reshape(
                    inp_size, inp_size, len(set_id)) / pamp
                # pred_size = int(np.sqrt(self.n_pred[pc_i]))
                reconst_img = (self.w['pc' + str(pc_i + 1)].numpy() @
                               self.xtr_record[sum(self.neurons_per_group[:3 * (pc_i + 1)]):
                                               sum(self.neurons_per_group[:3 * (pc_i + 1)])
                                               + self.n_pred[pc_i]]).numpy()[:, set_id].reshape(inp_size,
                                                                                                inp_size,
                                                                                                len(set_id)) / pamp

                for plt_idx in range(len(set_id)):
                    input_plot = axs[plt_idx, 0 + pc_i * 3].imshow(input_img[:, :, plt_idx],
                                                                             cmap='Reds', vmin=1000, vmax=3000)
                    fig.colorbar(input_plot, ax=axs[plt_idx, 0 + pc_i * 3], shrink=0.6)
                    reconst_plot = axs[plt_idx, 1 + pc_i * 3].imshow(reconst_img[:, :, plt_idx],
                                                                               cmap='Reds', vmin=1000, vmax=3000)
                    fig.colorbar(reconst_plot, ax=axs[plt_idx, 1 + pc_i * 3], shrink=0.6)
                    diff_plot = axs[plt_idx, 2 + pc_i * 3].imshow(
                        input_img[:, :, plt_idx] - reconst_img[:, :, plt_idx],
                        cmap='bwr',
                        vmin=-1000, vmax=1000)
                    fig.colorbar(diff_plot, ax=axs[plt_idx, 2 + pc_i * 3], shrink=0.6)

            [axi.axis('off') for axi in axs.ravel()]
            fig.suptitle('progress update: epoch #{0}/{1}'.format(epoch_i + 1 + prg_start_idx,
                                                                  num_epoch + prg_start_idx))
            fig.savefig(self.model_dir + '/progress_update_{0:0=2d}.png'.format(epoch_i + 1 + prg_start_idx))
            plt.close(fig)

        # time remaining
        epoch_time_avg += time.time() - epoch_time
        update_sim_time(self.model_dir, '\n***** time remaining = {0}'.format(
            str(timedelta(seconds=epoch_time_avg / (epoch_i + 1) * (num_epoch - epoch_i - 1)))))

        sse_fig, sse_axs = plt.subplots(nrows=self.n_pc_layer, ncols=1, sharex=True)
        for i in range(1, self.n_pc_layer + 1):
            bu_start_idx = sum(self.neurons_per_group[:3 * (i - 1)])
            bu_end_idx = bu_start_idx + self.neurons_per_group[3*(i-1)]
            td_start_idx = sum(self.neurons_per_group[:3 * i])
            td_end_idx = td_start_idx + self.n_pred[i - 1]

            bu_input = self.xtr_record[bu_start_idx:bu_end_idx] / pamp
            td_pred = (self.w['pc' + str(i)] @ self.xtr_record[td_start_idx:td_end_idx]) / pamp

            self.sse['pc' + str(i)].append(tf.reduce_sum(tf.reduce_mean((td_pred - bu_input) ** 2, axis=1)).numpy())
            sse_axs[i - 1].plot(np.arange(epoch_i + 1 + prg_start_idx), np.log(self.sse['pc' + str(i)]))
            sse_axs[i - 1].set_xlabel('epoch #')
            sse_axs[i - 1].set_ylabel('log (SSE)')
            sse_axs[i - 1].label_outer()

        sse_fig.suptitle('SSE update: epoch #{0}/{1}'.format(epoch_i + 1 + prg_start_idx,
                                                             num_epoch + prg_start_idx))
        sse_fig.savefig(self.model_dir + '/log_sse.png')#.format(epoch_i + 1 + prg_start_idx))
        plt.close(sse_fig)

        if (epoch_i+1+prg_start_idx) % n_plot_idx == 0:
            # weight dist change
            w_fig = weight_dist(savefolder=save_folder,
                                weights=self.w, weights_init=self.w_init,
                                n_pc=self.n_pc_layer, epoch_i=epoch_i + prg_start_idx)

            test_fig = self.test_inference(imgs=testing_set,
                                              nsample=test_n_sample, ndigit=n_class,
                                              simul_dur=simul_dur, sim_dt=sim_dt, sim_lt=sim_lt,
                                              train_or_test='test')

            # rdm analysis
            rdm_fig = self.rdm_plots(testing_current=testing_set, n_class=n_class,
                                     savefolder=self.model_dir, trained="test", epoch_i=epoch_i + prg_start_idx)

            self.save_results(epoch_i + prg_start_idx)

    end_time = time.time()
    update_sim_time(self.model_dir, '\nsimulation : {0}'.format(str(timedelta(seconds=end_time - start_time))))

    return self.sse

gpus = tf.config.experimental.list_logical_devices('GPU')
gpu_i = int(sys.argv[3])

with tf.device(gpus[gpu_i].name):

    AdEx_Layer.__init__ = new_init
    AdEx_Layer.train_network = new_train

    # # load constants
    # with open('adex_constants.pickle', 'rb') as f:
    #     AdEx = pickle.load(f)

    # specify the folder
    save_folder = sys.argv[1]

    # load sse from previous training
    with open(save_folder + '/sse_dict.pickle', 'rb') as sse_handle:
        sse_original = pickle.load(sse_handle)
    AdEx_Layer.sse = sse_original

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

    # load learned weights from previous training
    with open(save_folder + '/weight_init_dict.pickle', 'rb') as wdict:
        w_mat_init = pickle.load(wdict)
    # convert them to tensors
    for key, grp in w_mat_init.items():
        w_mat_init[key] = tf.convert_to_tensor(grp)

    # not necessary from next training
    n_epoch = int(sys.argv[2])

    # training_set, training_labels, test_set, test_labels, digits, training_set_idx
    training_set = np.load(save_folder + '/training_data.npy')
    test_set = np.load(save_folder + '/test_data.npy')

    n_stim = training_set.shape[1]
    sqrt_nstim = int(np.sqrt(n_stim))

    # test inference on test data
    test_n_shape = n_shape
    test_n_sample = 16
    test_iter_idx = int(n_samples/test_n_sample)

    testing_set = test_set[::test_iter_idx]

    # lrate = np.repeat(1.0, n_pc_layers) * 10 ** -10

    # build network
    adex_01 = AdEx_Layer(sim_directory=save_folder,
                         num_pc_layers=n_pc_layers,
                         num_pred_neurons=n_pred_neurons,
                         num_stim=n_stim,
                         gist_num=n_gist, w_mat=w_mat, w_mat_init=w_mat_init)
    # def new_init(self, sim_directory,
    #              # neuron_model_constants,
    #              num_pc_layers, num_pred_neurons, num_stim, gist_num,
    #              w_mat, w_mat_init):


    # train_network(self, num_epoch, sim_dur, sim_dt, sim_lt, lr, reg_a, input_current, n_shape, n_batch, set_idx):
    sse = adex_01.train_network(num_epoch=n_epoch,
                                simul_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                                lr=lrate, reg_a=reg_alpha,
                                input_current=training_set.T,
                                test_set=test_set, test_n_sample=16,
                                n_class=n_shape, batch_size=batch_size,
                                set_idx=rep_set_idx, report_idx=report_index, n_plot_idx=n_plot_idx)