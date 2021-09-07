import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle5 as pickle
from AdEx_const import *
from mnist_data import create_mnist_set

pamp = 10 ** -12

I_ext = np.arange(0, 3100, 100) * 10 ** -12
# I_ext = np.random.normal(loc=1000, scale=500, size=100) * 10 ** -12
n_stim = len(I_ext)
n_pred = len(I_ext)
# n_pred = 200
n_variable = n_stim + n_pred

# simulation parameters
T = 500 * 10 ** -3
dt = 1 * 10 ** -4

# synapse parameters
# w = tf.random.normal(shape=(200, 100), mean=1.0, stddev=0.5, dtype=tf.float32)
# w_clamp = tf.cast(tf.greater(w, 0), tf.float32)
# w = w * w_clamp
w_const = 550 * 10 ** -12
# w = w_const

# internal variables
v = tf.Variable(tf.ones([n_variable, ], dtype=tf.float32) * EL)
c = tf.Variable(tf.zeros([n_variable, ], dtype=tf.float32))
ref = tf.Variable(tf.zeros([n_variable, ], dtype=tf.float32))
# pre-synaptic variables
x = tf.Variable(tf.zeros([n_variable, ], dtype=tf.float32))
x_tr = tf.Variable(tf.zeros([n_variable, ], dtype=tf.float32))
# post-synaptic variable
Isyn = tf.Variable(tf.zeros([n_variable, ], dtype=tf.float32))
fired = tf.Variable(tf.zeros([n_variable, ], dtype=tf.float32))

# feed external corrent to the first layer
Iext = tf.constant(I_ext, dtype=tf.float32)
fs = tf.Variable(tf.zeros([n_variable, ], dtype=tf.float32))
xtr_s = tf.Variable(tf.zeros([int(T / dt)], dtype=tf.float32))

xxx = tf.Variable(tf.zeros([n_variable, int(T / dt)], dtype=tf.float32))

def update_var(v, c, ref, x, x_tr, Isyn, fs):
    # feed synaptic current to higher layers
    Isyn = update_Isyn(Isyn)

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
    fs = fs + fired
    # reset variables
    v = fired * EL + (1 - fired) * v
    c = fired * tf.add(c, b) + (1 - fired) * c
    x = fired * -x_reset + (1 - fired) * x

    # set lower boundary of v (Vrest = -70.6 mV)
    v = tf.maximum(EL, v)
    ref = tf.add(ref, fired * float(t_ref / dt))

    return v, c, ref, x, x_tr, Isyn, fs


def update_v(v, constraint):
    dv = (dt / Cm) * (gL * (EL - v) +
                      gL * DeltaT * tf.exp((v - VT) / DeltaT) +
                      Isyn - c)
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


def update_Isyn(Isyn):
    # I = ext
    Isyn[:n_stim].assign(Iext)
    Isyn[n_stim:].assign(x_tr[:n_stim] * w_const)
    # Isyn[n_stim:].assign(tf.reshape(w @ tf.reshape((x_tr[:n_stim] * w_const), shape=(n_stim, 1)), shape=(n_pred,)))

    return Isyn
    # # gist = W[ig]@ Isyn[I]
    # if _step < 500:
    #     input_gist = tf.transpose(w['ig']) @ (x_tr[:neurons_per_group[0]] * w_const)
    #     Isyn[-n_gist:, :].assign(input_gist)
    # else:
    #     Isyn[-n_gist:, :].assign(tf.zeros(shape=Isyn[-n_gist:, :].shape, dtype=tf.float32))
    #
    # for pc_layer_idx in range(n_pc_layer):
    #     Isyn_by_layer(pc_layer_idx)


# def Isyn_by_layer(self, pc_layer_idx):
#     # index of current prediction layer
#     curr_p_idx = sum(neurons_per_group[:pc_layer_idx * 3])
#     curr_p_size = neurons_per_group[pc_layer_idx * 3]
#
#     # index of next prediction layer
#     next_p_idx = sum(neurons_per_group[:pc_layer_idx * 3 + 3])
#     next_p_size = neurons_per_group[pc_layer_idx * 3 + 3]
#
#     # input / predictin error
#     bu_sensory = x_tr[curr_p_idx: curr_p_idx + curr_p_size, :] * w_const
#     # prediction
#     td_pred = w['pc' + str(pc_layer_idx + 1)] @ (
#             x_tr[next_p_idx:next_p_idx + next_p_size, :] * w_const)
#
#     # E+ = I - P
#     Isyn[curr_p_idx + curr_p_size:curr_p_idx + 2 * curr_p_size, :].assign(tf.add(bu_sensory, -td_pred))
#     # E- = -I + P
#     Isyn[curr_p_idx + 2 * curr_p_size:next_p_idx, :].assign(tf.add(-bu_sensory, td_pred))
#
#     # P = bu_error + td_error
#     bu_err_pos = tf.transpose(w['pc' + str(pc_layer_idx + 1)]) @ (
#             x_tr[curr_p_idx + curr_p_size:curr_p_idx + 2 * curr_p_size, :] * w_const)
#     bu_err_neg = tf.transpose(w['pc' + str(pc_layer_idx + 1)]) @ (
#             x_tr[curr_p_idx + 2 * curr_p_size:next_p_idx, :] * w_const)
#     gist = tf.transpose(w['gp' + str(pc_layer_idx + 1)]) @ (x_tr[-n_gist:, :] * w_const)
#
#     if pc_layer_idx < n_pc_layer - 1:
#         td_err_pos = x_tr[next_p_idx + next_p_size:next_p_idx + 2 * next_p_size] * w_const
#         td_err_neg = x_tr[next_p_idx + 2 * next_p_size:next_p_idx + 3 * next_p_size] * w_const
#         Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(
#             tf.add(
#                 tf.add(
#                     tf.add(bu_err_pos, -bu_err_neg),
#                     tf.add(-td_err_pos, td_err_neg)),
#                 gist))
#     else:
#         Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(tf.add(tf.add(bu_err_pos, -bu_err_neg), gist))


for t in range(int(T / dt)):
    # update internal variables (v, c, x, x_tr)
    v, c, ref, x, x_tr, Isyn, fs = update_var(v, c, ref, x, x_tr, Isyn, fired)
    xxx[:, t].assign(x_tr)

    # ) tf.Variable(tf.zeros([n_variable, int(T / dt)], dtype=tf.float32))


# # take the mean of synaptic output
# xtr_record.assign(xtr_record / (l_time / dt))
