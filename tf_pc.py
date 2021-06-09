import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

AdEx = {
    't_ref': 2 * 10 ** (-3),  # ms
    'Cm': 281 * 10 ** (-12),  # pF
    'gL': 30 * 10 ** (-9),  # nS=
    'EL': -70.6 * 10 ** (-3),  # mV
    'VT': -50.4 * 10 ** (-3),  # mV
    'DeltaT': 2 * 10 ** (-3),  # mV

    # Pick an electrophysiological behaviour
    # Regular spiking (as in the paper)
    'tauw': 144 * 10 ** (-3),  # ms
    'a': 4 * 10 ** (-9),  # nS
    'b': 0.0805 * 10 ** (-9),  # nA

    # spike trace
    'x_reset': 1.,
    'I_reset': -1 * 10 ** (-12),  # pamp
    'tau_rise': 5 * 10 ** (-3),  # ms
    'tau_s': 50 * 10 ** (-3)  # ms
}

sim_dur = 1000 * 10 ** (-3)  # ms
dt = 1 * 10 ** (-3)  # ms
num_neurons = 100
# Iext = 1200 * 10 ** -12  # pamp
Iext = tf.random.normal(shape=(num_neurons,), mean=1500 * 10 ** -12, stddev=500 * 10 ** -12, dtype=tf.float32,
                        name='external current')
time_steps = int(sim_dur / dt) + 1


def dvdt(vt, ct, isynt):
    return (dt / AdEx['Cm']) * (AdEx['gL'] * (AdEx['EL'] - vt) + AdEx['gL'] * AdEx['DeltaT'] * tf.exp(
        (vt - AdEx['VT']) / AdEx['DeltaT']) + isynt - ct)


def dcdt(vt, ct):
    return (dt / AdEx['tauw']) * (AdEx['a'] * (vt - AdEx['EL']) - ct)


def dxdt(xt):
    return dt * (-xt / AdEx['tau_rise'])


def dxtrdt(xt, xtrt):
    return dt * (-xt / AdEx['tau_rise'] - xtrt / AdEx['tau_s'])


tf_dvdt = tf.function(dvdt)
tf_dcdt = tf.function(dcdt)
tf_dxdt = tf.function(dxdt)
tf_dxtrdt = tf.function(dxtrdt)


class testLayer(tf.keras.layers.Layer):
    def __init__(self, n_neuron):
        super(testLayer, self).__init__()

        self.n_neuron = n_neuron
        self.time_step = dt

        self.v = tf.Variable(tf.ones(shape=(n_neuron,), dtype=tf.float32) * AdEx['EL'], name='membrane potential',
                             trainable='False')
        self.c = tf.Variable(tf.zeros(shape=(n_neuron,), dtype=tf.float32), name='adaptation variable',
                             trainable='False')
        self.ref = tf.Variable(tf.zeros(shape=n_neuron, dtype=tf.int32),
                               trainable='False')

    def call(self):
        for ti in range(self.time_step - 1):
            ref_op = tf.greater(ref, 0)
            v_ref = tf.where(ref_op, AdEx['EL'], v[:, ti])


# Define Sequential model with 3 layers
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, input_shape=(25,),
                              activation="relu", name="layer1"),
        tf.keras.layers.Dense(36, activation="relu", name="layer2"),
        tf.keras.layers.Dense(2, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((10, 25))
y = model(x)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.Input(shape=(16,)))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# # Now the model will take as input arrays of shape (None, 16)
# # and output arrays of shape (None, 32).
# # Note that after the first layer, you don't need to specify
# # the size of the input anymore:
# model.add(tf.keras.layers.Dense(32))
# model.output_shape
