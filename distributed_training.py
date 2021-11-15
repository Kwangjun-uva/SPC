import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist_data import create_mnist_set, plot_mnist_set
from tf_local import * #AdEx_Layer, pick_idx, conn_probs, save_data, update_sim_time
from datetime import datetime
import os
import sys
import time
import pickle5 as pickle

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
    np.savez(sim_name + '/test_dict',
             digits=classes,
             test_set_idx=training_set_idx,
             training_labels=test_labels)

    # save simulation params
    sim_params = {'n_pc_layers': n_pc_layers, 'n_pred_neurons': n_pred_neurons, 'n_gist': n_gist,
                  'batch_size': batch_size, 'n_samples': n_samples, 'n_shape': n_shape,
                  'sim_dur': sim_dur, 'dt': dt, 'learning_window': learning_window,
                  'n_epoch': n_epoch, 'lrate': lrate, 'reg_alpha': reg_alpha,
                  'report_index': report_index, 'n_plot_idx': n_plot_idx,
                  'conn_vals': conn_vals, 'max_vals': max_vals}
    with open(sim_name + '/sim_params_dict.pickle', 'wb') as handle2:
        pickle.dump(sim_params, handle2, protocol=pickle.HIGHEST_PROTOCOL)

def distributed_weight_update(self, lr, alpha_w):

        dws = {}

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

            dw_l1 = tf.cast(tf.greater(self.w['pc' + str(pc_layer_idx + 1)], 0.0), tf.float32)
            dw_mean_pos = tf.reduce_mean(dw_all_pos, axis=2) - alpha_w[pc_layer_idx] * dw_l1
            dw_mean_neg = tf.reduce_mean(dw_all_neg, axis=2) - alpha_w[pc_layer_idx] * dw_l1

            dw = tf.add(dw_mean_pos, -dw_mean_neg)

            dws['pc' + str(pc_layer_idx + 1)] = tf.nn.relu(tf.add(self.w['pc' + str(pc_layer_idx + 1)], dw))

        return dws

def distributed_training(self,
                         gpus,
                         num_epoch, simul_dur, sim_dt, sim_lt,
                         lr, reg_a,
                         input_current, test_current, test_nsample,
                         n_class, batch_size,
                         set_idx,
                         report_idx, n_plot_idx):

    plt.close('all')

    # create lr history
    self.lr_hist = {}
    for pc_i in range(1, self.n_pc_layer + 1):
        self.lr_hist['pc' + str(pc_i)] = []

    # number of batches
    n_batch = int(input_current.shape[1] / batch_size)

    start_time = time.time()

    # initialize epoch sim time average
    epoch_time_avg = 0

    for epoch_i in range(num_epoch):

        epoch_time = time.time()

        if ((epoch_i + 1) % 5 == 0) and (lr[0] > 1e-10):
            lr *= np.exp(-0.1 * ((epoch_i + 1) / 5))

        dw = {}
        for pc_i in range(1, self.n_pc_layer + 1):
            curr_pcl = 'pc' + str(pc_i)
            self.lr_hist[curr_pcl].append(lr[pc_i - 1])
            dw[curr_pcl] = tf.zeros(shape=self.w[curr_pcl].shape)

        # plot learning rate history
        lr_fig = plt.figure()
        plt.plot(self.lr_hist['pc1'])
        plt.xlabel('epoch #')
        plt.ylabel('learning rate')
        lr_fig.savefig(self.model_dir + '/lr_hist.png')

        for iter_i, gpu in enumerate(gpus):
            with tf.device(gpu.name):
\
                curr_batch = input_current[:, iter_i * batch_size:(iter_i + 1) * batch_size]

                self.__call__(sim_duration=simul_dur, time_step=sim_dt, lt=sim_lt,
                              I_ext=curr_batch,
                              bat_size=batch_size)

                # update weights per batch
                dws = self.weight_update(lr=lr, alpha_w=reg_a)
                # sum across gpus
                for key, grp in dw.items():
                    dw[key] = grp + dws[key]

        # average across all gpus
        for key, grp in dw.items():
            dw[key] = grp / len(gpus)
            # update weights
            self.w[key] += dw[key]

        end_sim_time = time.time()
        update_sim_time(self.model_dir,
                        '\nepoch #{0}/{1} = {2:.2f} sec'.format(epoch_i + 1, num_epoch,
                                                             end_sim_time - epoch_time))

        with tf.device('/CPU:0'):

            if ((epoch_i + 1) % report_idx == 0):  # and (len(set_idx) > iter_i):
                set_id = set_idx  # [iter_i]

                neurons_per_pc = self.neurons_per_group[::3]
                # plot progres : p1 - p3
                fig, axs = plt.subplots(ncols=3 * self.n_pc_layer, nrows=n_class,
                                        figsize=(4 * 3 * self.n_pc_layer, 4 * n_class))
                for pc_i in range(self.n_pc_layer):  # for a 3-PC model, 0, 1, 2
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
                        input_plot = axs[plt_idx, 0 + pc_i * 3].imshow(input_img[:, :, plt_idx], cmap='Reds', vmin=1000,
                                                                       vmax=3000)
                        fig.colorbar(input_plot, ax=axs[plt_idx, 0 + pc_i * 3], shrink=0.6)
                        reconst_plot = axs[plt_idx, 1 + pc_i * 3].imshow(reconst_img[:, :, plt_idx], cmap='Reds', vmin=1000,
                                                                         vmax=3000)
                        fig.colorbar(reconst_plot, ax=axs[plt_idx, 1 + pc_i * 3], shrink=0.6)
                        diff_plot = axs[plt_idx, 2 + pc_i * 3].imshow(input_img[:, :, plt_idx] - reconst_img[:, :, plt_idx],
                                                                      cmap='bwr',
                                                                      vmin=-1000, vmax=1000)
                        fig.colorbar(diff_plot, ax=axs[plt_idx, 2 + pc_i * 3], shrink=0.6)

                [axi.axis('off') for axi in axs.ravel()]
                fig.suptitle('progress update: epoch #{0}/{1}'.format(epoch_i + 1, num_epoch))
                fig.savefig(self.model_dir + '/progress_update_{0:0=2d}.png'.format(epoch_i + 1))
                plt.close(fig)
                plt.close('all')

            # time remaining
            epoch_time_avg += time.time() - epoch_time
            update_sim_time(self.model_dir, '\n***** time remaining = {0}'.format(
                str(timedelta(seconds=epoch_time_avg / (epoch_i + 1) * (num_epoch - epoch_i - 1)))))

            sse_fig, sse_axs = plt.subplots(nrows=self.n_pc_layer, ncols=1, sharex=True)
            for i in range(1, self.n_pc_layer + 1):
                bu_start_idx = sum(self.neurons_per_group[:3 * (i - 1)])
                bu_end_idx = bu_start_idx + self.neurons_per_group[3 * (i - 1)]
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
            sse_fig.savefig(self.model_dir + '/log_sse.png'.format(epoch_i + 1))
            plt.close(sse_fig)

            if (epoch_i == 0) or ((epoch_i + 1) % n_plot_idx == 0):
                # weight dist change
                w_fig = weight_dist(savefolder=self.model_dir,
                                    weights=self.w, weights_init=self.w_init,
                                    n_pc=self.n_pc_layer, epoch_i=epoch_i)

                test_fig = self.test_inference(data_set=test_current,
                                               ndigit=n_class, nsample=test_nsample,
                                               simul_dur=simul_dur, sim_dt=sim_dt, sim_lt=sim_lt,
                                               train_or_test='test')

                # rdm analysis
                rdm_fig = self.rdm_plots(testing_current=test_current, n_class=n_class,
                                         savefolder=self.model_dir, trained="test", epoch_i=epoch_i)

            self.save_results(epoch_i)

        end_time = time.time()
        update_sim_time(self.model_dir, '\nsimulation : {0}'.format(str(timedelta(seconds=end_time - start_time))))

    return self.sse

# redefine functions
AdEx_Layer.weight_update = distributed_weight_update
AdEx_Layer.train_network = distributed_training

# network parameters
n_pred_neurons = [36 ** 2, 34 ** 2, 32 ** 2] # preferably each entry is an integer that has an integer square root
n_pc_layers = len(n_pred_neurons)
n_gist = 128

# create external input
batch_size = 256
n_shape = 10
n_samples = 512

# simulate
sim_dur = 350 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-4)  # ms
learning_window = 100 * 10 ** -3
report_index = 1

n_epoch = int(sys.argv[2])
lrate = np.repeat(1.0, n_pc_layers) * 10 ** -7
reg_alpha = np.repeat(1.0, n_pc_layers) * 10 ** -10

gpus = tf.config.list_logical_devices('GPU')

if sys.argv[1] == 'mnist':
    keras_data = tf.keras.datasets.mnist
elif sys.argv[1] == 'fmnist':
    keras_data = tf.keras.datasets.fashion_mnist

training_set, training_labels, test_set, test_labels, classes, training_set_idx = create_mnist_set(
    data_type=keras_data,
    nDigit=n_shape,
    nSample=n_samples,
    shuffle=True)
n_stim = training_set.shape[1]
sqrt_nstim = int(np.sqrt(n_stim))

rep_set_idx = pick_idx(training_labels, classes, batch_size)
n_plot_idx = 10

conn_vals = np.array([conn_probs(a_i, b_i)
                      for a_i, b_i in zip([n_stim] + [n_gist] * n_pc_layers, [n_gist] + n_pred_neurons)]) * 0.05

max_vals = np.array([1] * (n_pc_layers + 1)) * 0.25 * 5

# test inference on test data
test_n_shape = n_shape
test_n_sample = 16
test_iter_idx = int(n_samples / test_n_sample)

testing_set = test_set[::test_iter_idx]

# create a folder to save results
save_folder = 'distributed_nD' + str(n_shape) + 'nS' + str(n_samples) + 'nEP' + str(n_epoch)
if os.path.exists(save_folder):
    save_folder += datetime.today().strftime('_%Y_%m_%d_%H_%M')
os.mkdir(save_folder)

save_data(save_folder)

# print how many GPUs
update_sim_time(save_folder, "Num GPUs Available: {0}\n".format(tf.config.list_physical_devices('GPU')))

# plot the same test set
plot_mnist_set(testset=training_set, testset_idx=training_set_idx,
               nDigit=n_shape, nSample=n_samples,
               savefolder=save_folder)

# build network
adex_01 = AdEx_Layer(sim_directory=save_folder,
                     # neuron_model_constants=AdEx,
                     num_pc_layers=n_pc_layers,
                     num_pred_neurons=n_pred_neurons,
                     num_stim=n_stim,
                     gist_num=n_gist, gist_connp=conn_vals, gist_maxw=max_vals)

# train_network
sse = adex_01.train_network(num_epoch=n_epoch,
                            gpus=gpus,
                            simul_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                            lr=lrate, reg_a=reg_alpha,
                            input_current=training_set.T,
                            test_current=testing_set, test_nsample=test_n_sample,
                            n_class=n_shape, batch_size=batch_size,
                            set_idx=rep_set_idx, report_idx=report_index, n_plot_idx=n_plot_idx)

# save simulation data
save_data(save_folder)