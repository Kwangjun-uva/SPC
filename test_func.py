import numpy as np
import matplotlib.pyplot as plt

import create_33images


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

def weight_dist(weights, weights_init, n_stim):

    init_w = weights_init[n_stim:2*n_stim, 3*n_stim:].numpy().flatten()
    updated_w = weights[n_stim:2*n_stim, 3*n_stim:].numpy().flatten()
    nonzero_w = updated_w[updated_w>0]

    fig = plt.figure()
    plt.subplot(211)
    plt.hist(init_w, bins=100)
    plt.title('initial weight distribution')
    plt.subplot(212)
    plt.hist(nonzero_w, bins=100,
             label='nonzero W = {0:.2f}%'.format(len(nonzero_w) / len(updated_w) * 100))
    plt.title('weight distribution')
    plt.legend()

    return fig