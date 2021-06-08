from brian2 import *
import matplotlib.pyplot as plt

# neuron parameters : Adaptive exponential integrate-and-fire model (Brette and Gerstner, 2005)
int_method = 'euler'
syn_cal_method = 'euler'
t_ref = 2 * ms
Cm = 281 * pF
gL = 30 * nS
taum = Cm / gL
EL = -70.6 * mV
VT = -50.4 * mV
DeltaT = 2 * mV
Vcut = VT + 5 * DeltaT

# Pick an electrophysiological behaviour
tauw, a, b, Vr = 144 * ms, 4 * nS, 0.0805 * nA, -70.6 * mV  # Regular spiking (as in the paper)
#tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
#tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking

# spike trace
x_reset = 1.
I_reset = -1 * pamp
tau_rise = 5 * ms
tau_s = 50 * ms

# at the time of spike
thres_cond = "v > VT"
reset_cond = "v = Vr; c+=b; x_up=-x_reset"

pre_eq = 'Iup = I_reset'

# number of neurons to simulate
n_neurons = 32
defaultclock.dt = 1 * ms
# neuron equations : Brette and Gerstner, 2005
eq_pre = """
dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + Isynapse - c)/Cm : volt
dc/dt = (a*(v - EL) - c)/tauw : amp

dx_up/dt    = - x_up/tau_rise : 1
dx_trace/dt = - x_up/tau_rise - x_trace/tau_s  : 1

Isynapse : amp
"""

eq_post = """
dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + Isynapse - c)/Cm : volt
dc/dt = (a*(v - EL) - c)/tauw : amp

dx_up/dt    = - x_up/tau_rise :1
dx_trace/dt = - x_up/tau_rise - x_trace/tau_s  : 1

Isynapse = Isyn_in_out : amp
Isyn_in_out : amp
"""

# synapse equation
eq_syn = '''
dItot/dt = -Iup/tau_rise - Itot/tau_s : amp (clock-driven)
dIup/dt = -Iup/tau_rise : amp (clock-driven)
w: 1

Isyn_in_out_post = w * Itot : amp (summed)
'''

# create neuron groups
ng_pre = NeuronGroup(N=16,
                     model=eq_pre,
                     threshold=thres_cond,
                     reset=reset_cond,
                     refractory=t_ref,
                     method=int_method)

ng_post = NeuronGroup(N=16,
                      model=eq_post,
                      threshold=thres_cond,
                      reset=reset_cond,
                      refractory=t_ref,
                      method=int_method)

# set initial membran potential
ng_pre.v = EL
ng_post.v = EL

syn_pre_post = Synapses(source=ng_pre,
                        target=ng_post,
                        model=eq_syn,
                        on_pre=pre_eq,
                        method=syn_cal_method)

# connect pre and post
syn_pre_post.connect('i==j') # one-to-one connection
# syn_pre_post.connect() # fully connect

# specify the weight between pre and post
syn_pre_post.w = 550
# specify the constant input current to the presynpatic neuron
input_curr = np.arange(0, 1600, 100)
ng_pre.Isynapse = input_curr * pamp

# variable monitors
monitor_pre = StateMonitor(source=ng_pre,
                           variables=['Isynapse', 'x_trace', 'v'],
                           record=True)
monitor_post = StateMonitor(source=ng_post,
                           variables=['Isynapse', 'Isyn_in_out', 'x_trace', 'v'],
                           record=True)

# spike monitors
spike_pre = SpikeMonitor(source=ng_pre)
spikes_post = SpikeMonitor(source=ng_post)

# run simulation
sim_dur = 1000 * ms
run(sim_dur)

# synaptic output from Pre (estimated with spike trace)
pre_output = monitor_pre.x_trace[0] * syn_pre_post.w[0]
# synaptic input to Post (recorded at Post)
post_input = monitor_post.Isynapse[0]/pamp
# membrane potential of Pre
pre_vm = monitor_pre.v[0]/mV

fig, axs = plt.subplots(ncols=2, nrows=1,
                        figsize=(2*4, 4))

axs[0].scatter(input_curr, np.array([np.mean(monitor_pre.x_trace[i, -200:]) for i in range(16)]) * syn_pre_post.w)
axs[1].scatter(input_curr, np.array([np.mean(monitor_post.Isynapse[i, -200:] / pamp) for i in range(16)]))

plt.show()

# # plot
# fig, axes = plt.subplots(ncols=1, nrows=5, sharex=True, squeeze=True)
#
# axes[0].plot(pre_vm)
# axes[0].set_ylabel('mV')
# axes[0].set_title('membrane potential of Pre')
#
# axes[2].plot(pre_output,
#              label='mean = {0:.2f}'.format(pre_output[-200:].mean()))
# axes[2].set_ylabel('pamp')
# axes[2].legend(loc='lower right')
# axes[2].set_title('synaptic output from Pre (estimated with spike trace)')
#
# axes[3].plot(post_input,
#              label='mean = {0:.2f}'.format(post_input[-200:].mean()))
# axes[3].set_ylabel('pamp')
# axes[3].set_title('synaptic input to Post (recorded at Post)')
# axes[3].legend(loc='lower right')
#
# axes[4].plot(monitor_post.v[0] / mV)
# axes[4].set_xlabel('time (ms)')
# axes[4].set_ylabel('mV')
# axes[4].set_title('membrane potential of Post')
#
# axes[1].eventplot(spike_pre.t/ms * (1 * ms) / defaultclock.dt)
# axes[1].set_yticks([])
# axes[1].set_title('spike train of Pre')
#
#
# for ax in axes:
#     ax.label_outer()
#
# plt.tight_layout()
# plt.show()