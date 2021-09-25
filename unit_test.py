import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle5 as pickle
from AdEx_const import *
from mnist_data import create_mnist_set, scale_tensor

I_ext = np.arange(0, 5100, 100) * pamp
# I_ext = np.random.normal(loc=1000, scale=500, size=100) * 10 ** -12
n_stim = len(I_ext)
n_pred = len(I_ext)
# n_pred = 200
n_variable = n_stim + n_pred

# simulation parameters
T = 350 * 10 ** -3
dt = 1 * 10 ** -4

# synapse parameters
# x_reset = 400 * 10 ** (-12)
# offset = 600 * 10 ** (-12)

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
xtr_s = tf.Variable(tf.zeros([n_variable, int(T / dt)], dtype=tf.float32))
xxx = tf.Variable(tf.zeros([n_variable, int(T / dt)], dtype=tf.float32))
Isyn_s = tf.Variable(tf.zeros([n_variable, int(T / dt)], dtype=tf.float32))

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
    fs.assign_add(fired)
    # reset variables
    v = fired * EL + (1 - fired) * v
    c = fired * tf.add(c, b) + (1 - fired) * c
    x = fired * -x_reset + (1 - fired) * x
    # x_tr = update_xtr(x_tr)

    # set lower boundary of v (Vrest = -70.6 mV)
    v = tf.maximum(EL, v)
    ref = tf.add(ref, fired * float(t_ref / dt))

    return v, c, ref, x, x_tr, Isyn, fs


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


def update_Isyn(Isyn):
    # I = ext
    Isyn[:n_stim].assign(Iext)
    # Isyn[n_stim:].assign(x_tr[:n_stim] * w_const)
    Isyn[n_stim:].assign(x_tr[:n_stim])
    # Isyn[n_stim:].assign(tf.reshape(w @ tf.reshape((x_tr[:n_stim] * w_const), shape=(n_stim, 1)), shape=(n_pred,)))

    return Isyn

for t in range(int(T / dt)):
    # update internal variables (v, c, x, x_tr)
    v, c, ref, x, x_tr, Isyn, fs = update_var(v, c, ref, x, x_tr, Isyn, fired)
    xtr_s[:, t].assign(x_tr)
    # xxx[:, t].assign(x)
    Isyn_s[:, t].assign(Isyn)

avg_time = int(100e-3 * 1/dt)
plt.figure()
plt.subplot(121)
plt.scatter(I_ext/pamp, Isyn_s[n_stim:, -avg_time:].numpy().mean(axis=1)/pamp, label='post_input')
plt.scatter(I_ext/pamp, xtr_s[:n_stim, -avg_time:].numpy().mean(axis=1)/pamp, c='g', label='pre_output')
plt.plot(I_ext/pamp, I_ext/pamp, c='r', ls='--')
plt.xlabel('input current to pre-syn neuron (pamp)')
plt.ylabel('current at post-syn terminal (pamp)')
plt.legend()

plt.subplot(122)
plt.scatter(I_ext/pamp, fs[:n_stim].numpy())
plt.xlabel('input current to pre-syn neuron (pamp)')
plt.ylabel('firing rate of pre-syn neuron (Hz)')
plt.show()

x_idx = 22
plt.figure()
# plt.plot(xxx[x_idx]/pamp)
plt.plot(Isyn_s[x_idx + n_stim]/pamp, c='b', label='post_input={0:.2f}'.format(Isyn_s[x_idx + n_stim, -avg_time:].numpy().mean()/pamp))
plt.plot(xtr_s[x_idx]/pamp, c='r', label='pre_input={0:.2f}\npre_output={1:.2f}'.format(I_ext[x_idx]/pamp, xtr_s[x_idx, -avg_time:].numpy().mean()/pamp))
plt.legend()
plt.show()

# def x_func(s, aa, x, xtr, dt=0.0001):
#     for i in range(len(s)):
#         if s[i] == 1:
#             x.append(-aa)
#         else:
#             x.append(x[i] + dt * (-x[i] / 0.005))
#         xtr.append(xtr[i] + dt * (-x[i] / 0.005 - xtr[i] / 0.05))
#     return xtr