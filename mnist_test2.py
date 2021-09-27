import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import pickle5 as pickle
from AdEx_const import *
from mnist_data import create_mnist_set
from create_33images import *

# def poisson_spikegen(fr, tsim):
#     dt = 1e-4
#     nbins = int(tsim/dt)
#     spikemat = (np.random.uniform(0, 1, nbins) < fr * dt).astype(float)
#
#     return spikemat

# shapes = create_shapes(base_mean=1500 * pamp,
#                        noise_var=500 * pamp,
#                        n_imgs_per_shape=10)

n_sample = 256
n_digit = 10
training_set, training_labels, test_set, test_labels, digits, training_set_idx = create_mnist_set(
    data_type=tf.keras.datasets.mnist,
    nSample=n_sample, nDigit=n_digit)



batch_size = n_sample * n_digit

Iext = training_set.T

n_stim = Iext.shape[0]
n_err = n_stim
n_pred = [900, 625]
n_pc_layer = len(n_pred)
neurons_per_group = [n_stim] * 3 + [n_pred[i] for i in range(len(n_pred) - 1)] * 3 + [n_pred[-1]]
n_variable = sum(neurons_per_group)

# simulation parameters
T = 350 * 10 ** -3
dt = 1 * 10 ** -4
lt = 100 * 10 ** -3

# synapse parameters
w_init = {}
for pc_i in range(n_pc_layer):
    w_init['pc' + str(pc_i + 1)] = np.abs(np.random.normal(loc=0.0, scale=0.025,
                                                    size=(neurons_per_group[3 * pc_i], neurons_per_group[3 * pc_i + 3])))

w = {}
for key, grp in w_init.items():
    w[key] = tf.constant(np.copy(grp), dtype=tf.float32)


def init_var():
    # internal variables
    v = tf.Variable(tf.ones([n_variable, batch_size], dtype=tf.float32) * EL)
    c = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))
    ref = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))
    # pre-synaptic variables
    x = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))
    x_tr = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))

    # post-synaptic variable
    Isyn = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))
    fired = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))

    return v, c, ref, x, x_tr, Isyn, fired


# fs = tf.Variable(tf.zeros([n_variable, ], dtype=tf.float32))
# xtr_s = tf.Variable(tf.zeros([n_variable, int(T / dt)], dtype=tf.float32))
# xxx = tf.Variable(tf.zeros([n_variable, int(T / dt)], dtype=tf.float32))
# Isyn_s = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))


def update_var(v, c, ref, x, x_tr, Isyn, fired, w):
    # feed synaptic current to higher layers
    Isyn = update_Isyn(Isyn, w)

    # current refractory status [0,2] ms
    ref_constraint = tf.cast(tf.greater(ref, 0), tf.float32)
    # update v according to ref: if in ref, dv = 0
    v = update_v(v, ref_constraint)
    c = update_c(c, ref_constraint)

    # subtract one time step (1) from refractory vector
    ref = tf.cast(tf.maximum(tf.subtract(ref, 1), 0), tf.float32)

    # update synaptic current
    x = update_x(x)
    x_tr = update_xtr(x_tr)

    # update spike monitor (fired: dtype=bool): if fired = True, else = False
    fired = tf.cast(tf.greater_equal(v, VT), tf.float32)
    # fs.assign_add(fired)

    # reset variables
    v = fired * EL + (1 - fired) * v
    c = fired * tf.add(c, b) + (1 - fired) * c
    x = fired * -x_reset + (1 - fired) * x
    # x_tr = update_xtr(x_tr)

    # set lower boundary of v (Vrest = -70.6 mV)
    v = tf.maximum(EL, v)
    ref = tf.add(ref, fired * float(t_ref / dt))

    return v, c, ref, x, x_tr, Isyn


def update_v(v, constraint):
    dv = (dt / Cm) * (gL * (EL - v) +
                      gL * DeltaT * tf.exp((v - VT) / DeltaT) +
                      (Isyn + offset) - c)
    dv_ref = (1 - constraint) * dv
    return tf.add(v, dv_ref)


def update_c(c, constraint):
    dc = (dt / tauw) * (a * (v - EL) - c)
    dc_ref = (1 - constraint) * dc
    return tf.add(c, dc_ref)


def update_x(x):
    dx = dt * (-x / tau_rise)
    return tf.add(x, dx)


def update_xtr(x_tr):
    dxtr = dt * (-x / tau_rise - x_tr / tau_s)
    return tf.add(x_tr, dxtr)


def update_Isyn(Isyn, w):
    # I = ext
    Isyn[:n_stim].assign(Iext)

    for pc_layer_idx in range(n_pc_layer):

        # index of current prediction layer
        curr_p_idx = sum(neurons_per_group[:pc_layer_idx * 3])
        curr_p_size = neurons_per_group[pc_layer_idx * 3]

        # index of next prediction layer
        next_p_idx = sum(neurons_per_group[:pc_layer_idx * 3 + 3])
        next_p_size = neurons_per_group[pc_layer_idx * 3 + 3]

        # input / prediction error
        bu_sensory = x_tr[curr_p_idx: curr_p_idx + curr_p_size, :]
        # prediction
        td_pred = w['pc' + str(pc_layer_idx + 1)] @ x_tr[next_p_idx:next_p_idx + next_p_size, :]

        # E+ = I - P
        Isyn[curr_p_idx + curr_p_size:curr_p_idx + 2 * curr_p_size, :].assign(
            tf.add(bu_sensory, -td_pred))
        # E- = -I + P
        Isyn[curr_p_idx + 2 * curr_p_size:next_p_idx, :].assign(
            tf.add(-bu_sensory, td_pred))

        # P = bu_error + td_error + gist
        bu_err_pos = tf.transpose(w['pc' + str(pc_layer_idx + 1)]) @ \
                     x_tr[curr_p_idx + curr_p_size:curr_p_idx + 2 * curr_p_size, :]
        bu_err_neg = tf.transpose(w['pc' + str(pc_layer_idx + 1)]) @ \
                     x_tr[curr_p_idx + 2 * curr_p_size:next_p_idx, :]
        # gist = tf.transpose(w['gp' + str(pc_layer_idx + 1)]) @ x_tr[-self.n_gist:, :]

        if pc_layer_idx < n_pc_layer - 1:
            td_err_pos = x_tr[next_p_idx + next_p_size:next_p_idx + 2 * next_p_size]
            td_err_neg = x_tr[next_p_idx + 2 * next_p_size:next_p_idx + 3 * next_p_size]
            Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(
                tf.add(
                    tf.add(bu_err_pos, -bu_err_neg),
                    tf.add(-td_err_pos, td_err_neg)
                )
            )
            # gist))
        else:
            Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(
                tf.add(
                    bu_err_pos, -bu_err_neg)
            )
            # gist))

    return Isyn

def weight_update(w, lr, alpha_w):
    new_w = {}

    for pc_layer_idx in range(n_pc_layer):
        err_idx = sum(neurons_per_group[:pc_layer_idx * 3 + 1])
        err_size = neurons_per_group[pc_layer_idx * 3 + 1]
        pred_idx = sum(neurons_per_group[:pc_layer_idx * 3 + 3])
        pred_size = n_pred[pc_layer_idx]

        xtr_ep = x_tr_records[err_idx: err_idx + err_size]
        xtr_en = x_tr_records[err_idx + err_size: err_idx + 2 * err_size]
        xtr_p = x_tr_records[pred_idx: pred_idx + pred_size]

        dw_all_pos = lr[pc_layer_idx] * tf.einsum('ij,kj->ikj', xtr_ep / pamp, xtr_p / pamp)
        dw_all_neg = lr[pc_layer_idx] * tf.einsum('ij,kj->ikj', xtr_en / pamp, xtr_p / pamp)

        dw_l1 = tf.cast(tf.greater(w['pc' + str(pc_layer_idx + 1)], 0.0), tf.float32)
        dw_mean_pos = tf.reduce_mean(dw_all_pos, axis=2) - alpha_w * dw_l1
        dw_mean_neg = tf.reduce_mean(dw_all_neg, axis=2) - alpha_w * dw_l1

        dws = tf.add(dw_mean_pos, -dw_mean_neg)
        new_w['pc' + str(pc_layer_idx + 1)] = tf.nn.relu(tf.add(w['pc' + str(pc_layer_idx + 1)], dws))

    return new_w

nsqrt = np.sqrt(np.array([n_stim] + n_pred)).astype(int)

n_epoch = 30
lr = [1e-7, 1e-7]
sse = {'pc1': [], 'pc2': []}
for i_epoch in range(n_epoch):
    print('epoch #{0}/{1}'.format(i_epoch + 1, n_epoch))

    v, c, ref, x, x_tr, Isyn, fired = init_var()
    x_tr_records = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))
    Isyn_s = tf.Variable(tf.zeros([n_variable, batch_size], dtype=tf.float32))
    for t in range(int(T / dt)):
        # update internal variables (v, c, x, x_tr)
        v, c, ref, x, x_tr, Isyn = update_var(v, c, ref, x, x_tr, Isyn, fired, w)
        if t > (lt/dt):
            x_tr_records.assign_add(x_tr)
            Isyn_s.assign_add(Isyn)
    x_tr_records = x_tr_records / ((T - lt)/dt)
    Isyn_s = Isyn_s / ((T - lt) / dt)

    inp1 = x_tr_records[:n_stim, ::n_sample] / pamp
    pred1 = (w['pc1'] @ x_tr_records[sum(neurons_per_group[:3]):sum(neurons_per_group[:3]) + n_pred[0]])[:, ::n_sample] / pamp
    # err1 = inp1 - pred1
    err1 = Isyn_s[n_stim:n_stim*2, ::n_sample] / pamp #- 600 * pamp
    inp2 = x_tr_records[sum(neurons_per_group[:3]):sum(neurons_per_group[:3]) + n_pred[0], ::n_sample] / pamp
    pred2 = (w['pc2'] @ x_tr_records[sum(neurons_per_group[:6]):sum(neurons_per_group[:6]) + n_pred[1]])[:, ::n_sample] / pamp
    err2 = Isyn_s[sum(neurons_per_group[:4]):sum(neurons_per_group[:5]), ::n_sample] / pamp #- 600 * pamp

    fig, axs = plt.subplots(nrows=n_digit, ncols=3 * n_pc_layer, figsize=(4 * 4, 4 * 3 * n_pc_layer))
    for j in range(n_digit):
        # col1 : input
        input_plot = axs[j, 0].imshow(inp1[:, j].numpy().reshape(nsqrt[0], nsqrt[0]), cmap="Reds")#, vmin=0, vmax=3000)
        fig.colorbar(input_plot, ax=axs[j, 0], shrink=0.2)
        # col2 : err
        err_plot = axs[j, 2].imshow(err1[:, j].numpy().reshape(nsqrt[0], nsqrt[0]), cmap="bwr", vmin=-1000, vmax=1000)
        fig.colorbar(err_plot, ax=axs[j, 2], shrink=0.2)
        # col3 : pred
        pred_plot = axs[j, 1].imshow(pred1[:, j].numpy().reshape(nsqrt[0], nsqrt[0]), cmap="Reds")#, vmin=0, vmax=3000)
        fig.colorbar(pred_plot, ax=axs[j, 1], shrink=0.2)
        # col1 : input
        input2_plot = axs[j, 3].imshow(inp2[:, j].numpy().reshape(nsqrt[1], nsqrt[1]), cmap="Reds")#, vmin=0, vmax=3000)
        fig.colorbar(input2_plot, ax=axs[j, 3], shrink=0.2)
        # col2 : err
        err2_plot = axs[j, 5].imshow(err2[:, j].numpy().reshape(nsqrt[1], nsqrt[1]), cmap="bwr", vmin=-1000, vmax=1000)
        fig.colorbar(err2_plot, ax=axs[j, 5], shrink=0.2)
        # col3 : pred
        pred2_plot = axs[j, 4].imshow(pred2[:, j].numpy().reshape(nsqrt[1], nsqrt[1]), cmap="Reds")#, vmin=0, vmax=3000)
        fig.colorbar(pred2_plot, ax=axs[j, 4], shrink=0.2)

    [axi.axis('off') for axi in axs.ravel()]
    fig.tight_layout()
    fig.show()

    if (i_epoch + 1) % 5 == 0:
        lr = [lr[pc_i] * (sse['pc' + str(pc_i + 1)][i_epoch - 1] / np.max(sse['pc' + str(pc_i + 1)]))
                for pc_i in range(n_pc_layer)]

    w = weight_update(w, lr=lr, alpha_w=1e-12)
    # dws = weight_update(w, lr=2e-7, alpha_w=1e-3)
    # w = tf.nn.relu(tf.add(w, dws))
    sse['pc1'].append(np.log(tf.reduce_sum(tf.reduce_mean(err1 ** 2)).numpy()))
    sse['pc2'].append(np.log(tf.reduce_sum(tf.reduce_mean(err2 ** 2)).numpy()))

    if (i_epoch + 1) % 10 == 0:
        fig2, axs2 = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        axs2[0, 0].hist(w_init['pc1'].flatten())
        axs2[1, 0].hist(w['pc1'].numpy().flatten())
        axs2[0, 1].hist(w_init['pc2'].flatten())
        axs2[1, 1].hist(w['pc2'].numpy().flatten())
        fig2.show()

        fig3, axs3 = plt.subplots(nrows=2, ncols=1, sharex=True)
        axs3[0].plot(sse['pc1'])
        axs3[1].plot(sse['pc2'])
        # axs3.xlabel('epoch #')
        # axs3.ylabel('SSE')
        fig3.show()

    plt.close('all')

# mmm = 2.0
# max_vals = np.array([n_gist/n_stim] + [n_pred_neurons[i] / n_gist for i in range(n_pc_layers)]) * mmm