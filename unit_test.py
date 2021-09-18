import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle5 as pickle
from AdEx_const import *
from mnist_data import create_mnist_set, scale_tensor

I_ext = np.arange(600, 3030, 30) * 10 ** -12
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
w_const = 1 * 10 ** -12

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
                      (Isyn + 600 * pamp) - c)
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
    Isyn[:n_stim].assign(scale_tensor(Iext, target_max=3000*pamp))
    # Isyn[n_stim:].assign(x_tr[:n_stim] * w_const)
    Isyn[n_stim:].assign(x_tr[:n_stim] )
    # Isyn[n_stim:].assign(tf.reshape(w @ tf.reshape((x_tr[:n_stim] * w_const), shape=(n_stim, 1)), shape=(n_pred,)))

    return Isyn

for t in range(int(T / dt)):
    # update internal variables (v, c, x, x_tr)
    v, c, ref, x, x_tr, Isyn, fs = update_var(v, c, ref, x, x_tr, Isyn, fired)
    xtr_s[:, t].assign(x_tr)
    xxx[:, t].assign(x)
    Isyn_s[:, t].assign(Isyn)

plt.figure()
plt.subplot(121)
plt.scatter(I_ext/pamp, Isyn_s[:n_stim, -2000:].numpy().mean(axis=1)/pamp)
plt.xlabel('input current to pre-syn neuron (pamp)')
plt.ylabel('current at post-syn terminal (pamp)')

plt.subplot(122)
plt.scatter(I_ext/pamp, fs[:n_stim].numpy())
plt.xlabel('input current to pre-syn neuron (pamp)')
plt.ylabel('firing rate of pre-syn neuron (Hz)')
plt.show()

plt.figure()
plt.plot(xxx[22]/pamp)
plt.plot(xtr_s[22]/pamp, c='r', label='input={0:.2f}\nmean={1:.2f}'.format(I_ext[22]/pamp, xtr_s[22, -2000:].numpy().mean()/pamp))
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