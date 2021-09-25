import numpy as np
import tensorflow as tf
from mnist_data import create_mnist_set, plot_mnist_set
from tf_local import AdEx_Layer, pick_idx, conn_probs, update_sim_time
from datetime import timedelta, datetime
import os
import sys
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

# network parameters
n_pred_neurons = [36 ** 2, 34 ** 2, 32 ** 2] # preferably each entry is an integer that has an integer square root
n_pc_layers = len(n_pred_neurons)
n_gist = 128

# create external input
batch_size = 640
n_shape = 10
n_samples = 64

# simulate
sim_dur = 350 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-4)  # ms
learning_window = 100 * 10 ** -3
report_index = 1

n_epoch = 100
lrate = np.repeat(1.0, n_pc_layers) * 10 ** -9
reg_alpha = np.repeat(1.0, n_pc_layers) * 10 ** -12

gpus = tf.config.experimental.list_logical_devices('GPU')

gpu_i = int(sys.argv[2])

with tf.device(gpus[gpu_i].name):

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
    # n_plot_idx = 1

    conn_vals = np.array([conn_probs(a_i, b_i)
                          for a_i, b_i in zip([n_stim] + [n_gist] * n_pc_layers, [n_gist] + n_pred_neurons)]) * 0.05

    max_vals = np.array([1] * (n_pc_layers + 1)) * 0.25 * 5

    # test inference on test data
    test_n_shape = n_shape
    test_n_sample = 16
    test_iter_idx = int(n_samples / test_n_sample)

    testing_set = test_set[::test_iter_idx]

    # create a folder to save results
    save_folder = 'gpu' + str(gpu_i+1) + '_nD' + str(n_shape) + 'nS' + str(n_samples) + 'nEP' + str(n_epoch)
    if os.path.exists(save_folder):
        save_folder += datetime.today().strftime('_%Y_%m_%d_%H_%M')
    os.mkdir(save_folder)

    save_data(save_folder)

    # print how many GPUs
    update_sim_time(save_folder, "Num GPUs Available: {0}\n".format(tf.config.list_physical_devices('GPU')))

    # identify current GPU
    update_sim_time(save_folder, "currently running on " + gpus[gpu_i].name + '\n')

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
                                simul_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                                lr=lrate, reg_a=reg_alpha,
                                input_current=training_set.T,
                                test_current=testing_set, test_nsample=test_n_sample,
                                n_class=n_shape, batch_size=batch_size,
                                set_idx=rep_set_idx, report_idx=report_index, n_plot_idx=n_plot_idx)

    # save simulation data
    save_data(save_folder)