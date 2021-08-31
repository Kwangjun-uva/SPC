import numpy as np
import matplotlib.pyplot as plt
import os.path
from datetime import datetime


def error_raster(fr_array, n_stim, sample_neuron_idx):
    """
    This function checks if e+ and e- neurons fire at the same time.
    If the model is correctly implemented, two neurons should not fire at the same time.

    :param sample_neuron_idx: int. index of a sample neuron whose firing times will be plotted.
    :param n_stim: int. size of the input (flattened)
    :param fr_array: array. firing time expressed as a boolean array with size = (sim time step, num neurons, batch size)
    :return: 1. boolean (if there is any time step at which both + and - fired,
             2. Raster plot of a sample pair (+ and - for the same input neuron)
    """

    ep_fr = fr_array[:, n_stim:2*n_stim, 0]
    en_fr = fr_array[:, 2*n_stim:3*n_stim, 0]

    ep_fr_time_step, ep_neuron_idx = [np.argwhere(ep_fr > 0)[:, 0], np.argwhere(ep_fr > 0)[:, 1]]
    en_fr_time_step, en_neuron_idx = [np.argwhere(en_fr > 0)[:, 0], np.argwhere(en_fr > 0)[:, 1]]

    fig = plt.figure()
    plt.scatter(ep_fr_time_step, ep_neuron_idx, c='b', label='E+')
    plt.scatter(en_fr_time_step, en_neuron_idx, c='r', label='E=')
    plt.title('error neuron #{0} raster plot over time'.format(sample_neuron_idx))
    plt.xlabel('time (0.1ms)')
    plt.ylabel('neuron number')

    return (ep_fr + en_fr > 1).any(), fig

def weight_dist(savefolder, weights, weights_init, n_pc, epoch_i):

    fig, axs = plt.subplots(nrows=2, ncols=n_pc, figsize=(4 * n_pc, 5))
    for plt_idx, (key, grp) in enumerate(weights.items()):
        if 'pc' in key:
            init_w = weights_init[key].numpy().flatten()
            updated_w = grp.numpy().flatten()
            nonzero_w = updated_w[updated_w>0]

            # initial weight distribution
            axs[0, plt_idx].hist(init_w, bins=100)
            axs[0, plt_idx].set_title(key + ' initial weight distribution')
            # weight distribution after learning
            axs[1, plt_idx].hist(nonzero_w, bins=100,
                     label='nonzero W = {0:.2f}%'.format(len(nonzero_w) / len(updated_w) * 100))
            axs[1, plt_idx].set_title(key + ' weight distribution')
            axs[1, plt_idx].legend()
        else:
            pass

    fig.tight_layout()
    savefile_name = 'weight_dist_change_{:03d}'.format(epoch_i+1)
    # if os.path.isfile(savefolder + '/' + savefile_name + '.png'):
        # savefile_name += datetime.today().strftime('_%Y_%m_%d_%H_%M')
    fig.savefig(savefolder + '/' + savefile_name + '.png')

    return fig

# int_size = [28] + np.sqrt(adex_01.n_pred[:-1]).astype(int).tolist()
#
# p_reps = {}
# for pc_i in range(1, 5):
#     source_p_size = adex_01.n_pred[pc_i - 1]
#
#     curr_p_start_idx = sum(adex_01.neurons_per_group[:3 * pc_i])
#     curr_p_end_idx = sum(adex_01.neurons_per_group[:3 * pc_i]) + source_p_size
#
#     p_reps['pc' + str(pc_i)] = (adex_01.w['pc' + str(pc_i)] @ adex_01.xtr_record[curr_p_start_idx:curr_p_end_idx, :]).numpy().reshape(int_size[pc_i-1], int_size[pc_i-1], 48)
#
# inps = {}
# for pc_i in range(1, 5):
#     inps['pc' + str(pc_i)] = adex_01.xtr_record[
#                             sum(adex_01.neurons_per_group[:3 * (pc_i-1)]):sum(adex_01.neurons_per_group[:3 * (pc_i-1)]) +
#                                                                       int_size[pc_i - 1] ** 2].numpy().reshape(int_size[pc_i-1], int_size[pc_i-1], 48)
