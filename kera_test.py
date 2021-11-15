import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from AdEx_const import *
import pickle5 as pickle
from mnist_data import create_mnist_set
import time


# norm_factor = 1.0 * pred_size
        # for pc_layer_idx in range(self.n_pc_layer):
        #     err_size = self.neurons_per_group[pc_layer_idx * 3 + 1]
        #     pred_size = self.n_pred[pc_layer_idx]
        #
        #     norm_factor = 1.0 * pred_size
        #
        #     # self.w['pc' + str(pc_layer_idx + 1)] = tf.random.normal((err_size, pred_size), 1.0, 0.3) / norm_factor
        #     self.w['pc' + str(pc_layer_idx + 1)] = tf.nn.relu(tf.random.normal((err_size, pred_size), 0.0, 0.3) / norm_factor)
        #     self.w_init['pc' + str(pc_layer_idx + 1)] = self.w['pc' + str(pc_layer_idx + 1)]

class adex_layer(tf.keras.layers.Layer):
    def __init__(self, neuron_type, layer_size, batch_size, w, w_gist,
                 dt, sim_time, gist_time, learning_window):
        '''
        neuron_type = {input, gist, error+, error-, pred, pred_last}
        layer_size : int32
        batch_size : int32
        w : float32 = (nPred x nErr)
        w_gist : dict = {'ig': (nInput x nGist), 'gp': (nGist x nPred)}
        '''

        super(adex_layer, self).__init__()

        self.layer_size = layer_size
        self.batch_size = batch_size
        self.neuron_type = neuron_type
        self.w = w
        self.w_gist = w_gist

        self.dt = dt
        self.sim_time = sim_time
        self.gist_time = gist_time
        self.learning_window = learning_window

        # internal variables
        self.v = tf.ones([self.layer_size, self.batch_size], dtype=tf.float32) * EL
        self.c = tf.zeros([self.layer_size, self.batch_size], dtype=tf.float32)
        self.ref = tf.zeros([self.layer_size, self.batch_size], dtype=tf.float32)
        self.fired = tf.zeros([self.layer_size, self.batch_size], dtype=tf.bool)
        # pre-synaptic variables
        self.x = tf.zeros([self.layer_size, self.batch_size], dtype=tf.float32)
        self.x_tr = tf.zeros([self.layer_size, self.batch_size], dtype=tf.float32)
        # post-synaptic variable
        self.xtr_record = tf.zeros([self.layer_size, self.batch_size], dtype=tf.float32)
        self.Isyn = tf.zeros([self.layer_size, self.batch_size], dtype=tf.float32)

        self.time_step = 0


    def call(self, inputs):

        self.update_Isyn(inputs)
        self.update_var()

        if self.time_step >= int((self.sim_time - self.learning_window) / self.dt):
            self.xtr_record += self.x_tr

        if (self.time_step + 1) == (self.sim_time / self.dt):
            self.xtr_record /= int((self.sim_time - self.learning_window) / self.dt)

        self.time_step += 1

    def update_var(self):

        # current refractory status [0,2] ms
        ref_constraint = tf.cast(tf.greater(self.ref, 0), tf.float32)
        # update v according to ref: if in ref, dv = 0
        self.update_v(ref_constraint)
        self.update_c(ref_constraint)

        # subtract one time step (1) from refractory vector
        self.ref = tf.cast(tf.maximum(tf.subtract(self.ref, 1), 0), tf.float32)

        # update synaptic current
        self.update_x()
        self.update_xtr()

        # update spike monitor (fired: dtype=bool): if fired = True, else = False
        self.fired = tf.cast(tf.greater_equal(self.v, VT), tf.float32)
        # reset variables
        self.v = self.fired * EL + (1 - self.fired) * self.v
        self.c = self.fired * tf.add(self.c, b) + (1 - self.fired) * self.c
        self.x = self.fired * -x_reset + (1 - self.fired) * self.x

        # self.update_xtr()

        # set lower boundary of v (Vrest = -70.6 mV)
        self.v = tf.maximum(EL, self.v)
        self.ref = tf.add(self.ref, self.fired * float(t_ref / self.dt))

    def update_v(self, constraint):
        dv = (self.dt / Cm) * (gL * (EL - self.v) + gL * DeltaT * tf.exp((self.v - VT) / DeltaT) + self.Isyn - self.c)
        dv_ref = (1 - constraint) * dv
        self.v = tf.add(self.v, dv_ref)

    def update_c(self, constraint):
        dc = (self.dt / tauw) * (a * (self.v - EL) - self.c)
        dc_ref = (1 - constraint) * dc
        self.c = tf.add(self.c, dc_ref)

    def update_x(self):
        dx = self.dt * (-self.x / tau_rise)
        self.x = tf.add(self.x, dx)

    def update_xtr(self):
        dxtr = self.dt * (-self.x / tau_rise - self.x_tr / tau_s)
        self.x_tr = tf.add(self.x_tr, dxtr)

    def update_Isyn(self, inputs):

        if self.neuron_type == 'input':
            ''' input receives nExt x nBatch (Isyn) '''
            self.Isyn = inputs
        elif self.neuron_type == 'gist':
            ''' gist receives nInput x nBatch (xtr)'''
            self.Isyn = tf.transpose(self.w_gist['ig']) @ inputs

        if self.time_step > self.gist_time:

            if 'error' in self.neuron_type:
                ''' 
                error receives :
                    1. nPred(l) x  nBatch (xtr) : sensory input
                    2. nPred(l+1) x nBatch (xtr) : prediction 
                '''
                # sensory input
                bu_sensory = inputs[0]
                # prediction
                td_pred = self.w @ inputs[1]

                # E+ = I - P
                if self.neuron_type == 'error+':
                    self.Isyn = tf.add(bu_sensory, -td_pred) + 600 * pamp
                # E- = -I + P
                elif self.neuron_type == 'error-':
                    self.Isyn = tf.add(-bu_sensory, td_pred) + 600 * pamp
            elif 'pred' in self.neuron_type:
                '''
                pred receives:
                    1. nG x nBatch (xtr) : gist
                    2. nErr(l-1) x nBatch (xtr) : BU prediction error+
                    3. nErr(l-1) x nBatch (xtr) : BU prediction error-
                    4. nErr(l) x nBatch (xtr) : TD prediction error+
                    5. nErr(l) x nBatch (xtr) : TD prediction error-
                The last pred layer does not receive 4,5
                '''
                gist = tf.transpose(self.w_gist['gp']) @ inputs[0]
                bu_err_pos = tf.transpose(self.w) @ inputs[1]
                bu_err_neg = tf.transpose(self.w) @ inputs[2]

                # P = bu_error + td_error + gist
                if self.neuron_type == 'pred_last':
                    self.Isyn = tf.add(
                        tf.add(bu_err_pos, -bu_err_neg),
                        gist)
                else:
                    td_err_pos = inputs[3]
                    td_err_neg = inputs[4]

                    self.Isyn = tf.add(
                        tf.add(
                            tf.add(bu_err_pos, -bu_err_neg),
                            tf.add(-td_err_pos, td_err_neg)),
                        gist)

# load MNIST dataset
keras_data = tf.keras.datasets.mnist
# pre-processing
## training_set = (nSample (nSample x nDigit), nPixel)
training_set, training_labels, test_set, test_labels, classes, training_set_idx = create_mnist_set(data_type=keras_data,
                                                                                                   nDigit=10,
                                                                                                   nSample=10,
                                                                                                   shuffle=True)
mnist_input = tf.transpose(tf.constant(training_set, dtype=tf.float32))

# load pre-trained weights
with open('./2021_09_21_18_17_nD3nS512nEP100/weight_dict.pickle', 'rb') as f:
    weight_dict = pickle.load(f)
# define weights for gist
w_gist = {key:weight_dict[key] for key in ['ig', 'gp1']}

# simulation parameters
nBatch = 100
sim_dt = 1e-4
sim_time = 350e-3
gist_time = 50e-3
learning_window = 150e-3

# neuron_type, layer_size, batch_size, w, w_gist
class snn_pc(tf.keras.Model):

    def __init__(self):#, sim_time, sim_dt, gist_time, learning_window):
        super(snn_pc, self).__init__()
        self.input_layer = adex_layer(neuron_type='input',
                     layer_size=784, batch_size=nBatch,
                     w=1, w_gist=w_gist,
                     sim_time=sim_time, gist_time=gist_time, learning_window=learning_window, dt=sim_dt)
        self.gist_layer = adex_layer(neuron_type='gist',
               layer_size=144, batch_size=nBatch,
               w=weight_dict['pc1'], w_gist={'ig': weight_dict['ig'], 'gp': weight_dict['gp1']},
               sim_time=sim_time, gist_time=gist_time, learning_window=learning_window, dt=sim_dt)
        self.err0p_layer = adex_layer(neuron_type='error+',
               layer_size=784, batch_size=nBatch,
               w=weight_dict['pc1'], w_gist={'ig': weight_dict['ig'], 'gp': weight_dict['gp1']},
               sim_time=sim_time, gist_time=gist_time, learning_window=learning_window, dt=sim_dt)
        self.err0n_layer = adex_layer(neuron_type='error-',
               layer_size=784, batch_size=nBatch,
               w=weight_dict['pc1'], w_gist={'ig': weight_dict['ig'], 'gp': weight_dict['gp1']},
               sim_time=sim_time, gist_time=gist_time, learning_window=learning_window, dt=sim_dt)
        self.pred1_layer = adex_layer(neuron_type='pred',
               layer_size=900, batch_size=nBatch,
               w=weight_dict['pc1'], w_gist={'ig': weight_dict['ig'], 'gp': weight_dict['gp1']},
               sim_time=sim_time, gist_time=gist_time, learning_window=learning_window, dt=sim_dt)
        self.err1p_layer = adex_layer(neuron_type='error+',
               layer_size=900, batch_size=nBatch,
               w=weight_dict['pc2'], w_gist={'ig': weight_dict['ig'], 'gp': weight_dict['gp2']},
               sim_time=sim_time, gist_time=gist_time, learning_window=learning_window, dt=sim_dt)
        self.err1n_layer = adex_layer(neuron_type='error-',
               layer_size=900, batch_size=nBatch,
               w=weight_dict['pc2'], w_gist={'ig': weight_dict['ig'], 'gp': weight_dict['gp2']},
               sim_time=sim_time, gist_time=gist_time, learning_window=learning_window, dt=sim_dt)
        self.pred2_layer = adex_layer(neuron_type='pred_last',
               layer_size=625, batch_size=nBatch,
               w=weight_dict['pc2'], w_gist={'ig': weight_dict['ig'], 'gp': weight_dict['gp2']},
               sim_time=sim_time, gist_time=gist_time, learning_window=learning_window, dt=sim_dt)

    def call(self, inputs):
        self.input_layer(inputs),
        self.gist_layer(self.input_layer.x_tr),
        self.err0p_layer([self.input_layer.x_tr, self.pred1_layer.x_tr]),
        self.err0n_layer([self.input_layer.x_tr, self.pred1_layer.x_tr]),
        self.pred1_layer([self.gist_layer.x_tr,
                          self.err0p_layer.x_tr, self.err0n_layer.x_tr,
                          self.err1p_layer.x_tr, self.err1n_layer.x_tr]),
        self.err1p_layer([self.pred1_layer.x_tr, self.pred2_layer.x_tr]),
        self.err1n_layer([self.pred1_layer.x_tr, self.pred2_layer.x_tr]),
        self.pred2_layer([self.gist_layer.x_tr,
                          self.err1p_layer.x_tr, self.err1n_layer.x_tr])

snn_pc = snn_pc()
# strategy = tf.distribute.MirroredStrategy()
#
# # print ('# devices : {}'.format(strategy.num_replicas_in_sync))
# # batch_per_replica = 50
# # batch_size = batch_per_replica * strategy.num_replicas_in_sync
#
# for epoch_i in range(2):
#     start_time = time.time()
#     for time_step in range(int(sim_time/sim_dt)):
#         with strategy.scope():
#             snn_pc(mnist_input)
#
#
#     print('#epoch{0}, {1:.2f} ms simulation in real time : {2:.2f} sec'.format(
#         int(epoch_i + 1), int(sim_time/1e-3), time.time() - start_time))
#
# # plot a figure
# fig, axs = plt.subplots(nrows=10, ncols=4)
# for i in range(10):
#     axs[i, 0].imshow(snn_pc.input_layer.xtr_record[:, ::10][:, i].numpy().reshape(28,28) / pamp)
#     axs[i, 1].imshow((weight_dict['pc1'] @ snn_pc.pred1_layer.xtr_record)[:, ::10][:, i].numpy().reshape(28, 28) / pamp)
#     axs[i, 2].imshow(snn_pc.pred1_layer.xtr_record[:, ::10][:, i].numpy().reshape(30, 30) / pamp)
#     axs[i, 3].imshow((weight_dict['pc2'] @ snn_pc.pred2_layer.xtr_record)[:, ::10][:, i].numpy().reshape(30, 30) / pamp)
#
# axs[0,0].set_title('input')
# axs[0,1].set_title('P1 reconstructed')
# axs[0,2].set_title('P1 input')
# axs[0,3].set_title('P2 reconstructed')
#
# for ax in axs.flatten():
#     ax.axis('off')
#
# plt.show()