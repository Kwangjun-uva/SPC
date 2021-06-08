from typing import List

from brian2 import *
from sklearn.manifold import TSNE
from scipy import stats
import pandas as pd

# neuron parameters
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

x_reset = 1.
I_reset = -1 * pamp
tau_rise = 5 * ms
tau_s = 50 * ms
# ampConst = 10 ** (-12)

thres_cond = "v > VT"
reset_cond = "v = Vr; c+=b; x_up=-x_reset"

pre_eq = 'Iup = I_reset'

Wconst = 550

# SIM TIMES
runtime_ff = 200
runtime_pc = 500
last_t = 200

nDigit = 10
nSample = 10
niter = 1
nPR = 1

nlayerPC = 4

l1reg_alpha = 1000
l2reg_alpha = 5

es: List[str] = ['p', 'n']

# neuron equations
basic_eq = """
dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + Isynapse - c)/Cm : volt
dc/dt = (a*(v - EL) - c)/tauw : amp

dx_up/dt    = - x_up/tau_rise :1
dx_trace/dt = - x_up/tau_rise - x_trace/tau_s  : 1

"""

template_Err_model = """
Isynapse = Ierror : amp
Ierror = errsgn * (Iinput - Ipred) : amp

Iinput = Isyn_P(l)_E(l)err_sign : amp
Isyn_P(l)_E(l)err_sign : amp
Ipred = Isyn_P(l+1)_E(l)err_sign : amp
Isyn_P(l+1)_E(l)err_sign : amp
"""

template_Pred_model = """
Isynapse = Igist + Ierror_bu - Ierror_td : amp

Ierror_bu = Isyn_E(l-1)p_P(l) - Isyn_E(l-1)n_P(l) : amp
Ierror_td = Isyn_E(l)p_P(l) - Isyn_E(l)n_P(l) : amp
Igist = Isyn_G_P(l) : amp

Isyn_E(l-1)p_P(l) : amp
Isyn_E(l)p_P(l) : amp

Isyn_E(l-1)n_P(l) : amp
Isyn_E(l)n_P(l) : amp

Isyn_G_P(l) : amp
"""

# synapse equations
eq_syn = '''
dItot/dt = -Iup/tau_rise - Itot/tau_s : amp (clock-driven)
dIup/dt = -Iup/tau_rise : amp (clock-driven)
w: 1

Isyn_TARGET_post = w * Itot : amp (summed)
'''

def network_size(nlayerlist, nlpc):
    """
    :param nlayerlist: list
    specifies the number of neurons in each layer/unit
    The order is [nI, nPC1, nPC2, ... , nPCn, nG] where each number is : int
    :param nlpc:/ int
    is the number of PC layers (one layer = one Error and one Pred)
    :return: n_network_dict : list
    contains the number of neurons in each layer/unit of the network
    """

    # assign neuron numbers to input and E0 units
    n_network_dict = {'I': nlayerlist[0],
                      'E0p': nlayerlist[0], 'E0n': nlayerlist[0]
                      }
    # assign neuron numbers to PC layers
    if nlpc > 1:
        for pc_i in range(1, nlpc):
            n_network_dict['E' + str(pc_i) + 'p'] = nlayerlist[pc_i]
            n_network_dict['E' + str(pc_i) + 'n'] = nlayerlist[pc_i]
            n_network_dict['P' + str(pc_i)] = nlayerlist[pc_i]
    # the last PC layer does not have error unit
    n_network_dict['P' + str(nlpc)] = nlayerlist[-2]
    # assign neuron numbers to gist unit
    n_network_dict['G'] = nlayerlist[-1]

    return n_network_dict


def create_neuron_models(nlpc, base_model_eq, temp_pred, temp_err):
    """
    :param nlpc: int
    represents the number of PC layers (one layer = one Error and one Pred)
    :param base_model_eq: str
    is a base template for AdEX neuron model
    :param temp_pred: str
    is a base template for AdEx neuron model for prediction neurons
    :param temp_err: str
    is a base template for AdEx neuron model for error neurons
    :return model_dict: dict
    contains AdEx neuron model equation for each unit/layer in the network
    """

    layers = ['(l-1)', '(l)', '(l+1)']
    # es = ['p', 'n']

    model_dict = {'I': base_model_eq + "Isynapse = Iext : amp\nIext : amp"}
    # model_dict = {}
    dd_e = "Isynapse = Ierror : amp\n" \
           "Ierror = errsgn * (Iinput - Ipred) : amp\n\n" \
           "Iinput = Isyn_I_E0err_sign : amp\n" \
           "Isyn_I_E0err_sign : amp\n" \
           "Ipred = Isyn_P1_E0err_sign : amp\n" \
           "Isyn_P1_E0err_sign : amp"

    for esi in es:
        if esi == 'n':
            esr = '-1'
        else:
            esr = '1'
        model_dict['E0' + esi] = base_model_eq + dd_e.replace('err_sign', esi).replace('errsgn', esr)

    for pc_i in range(1, nlpc):

        model_dict['P' + str(pc_i)] = base_model_eq + temp_pred
        for esi in es:
            if esi == 'n':
                esr = '-1'
            else:
                esr = '1'
            model_dict['E' + str(pc_i) + esi] = base_model_eq + temp_err.replace('err_sign', esi).replace('errsgn', esr)

        for j, layer in enumerate(layers):
            model_dict['P' + str(pc_i)] = model_dict['P' + str(pc_i)].replace(layer, str(pc_i - 1 + j))
            for esi in es:
                model_dict['E' + str(pc_i) + esi] = model_dict['E' + str(pc_i) + esi].replace(layer, str(pc_i - 1 + j))

    dd_p = "Isynapse = Ierror_bu + Igist : amp \n" \
           "Ierror_bu = Isyn_E(l-1)p_P(l) - Isyn_E(l-1)n_P(l) : amp \n" \
           "Isyn_E(l-1)p_P(l) : amp \n" \
           "Isyn_E(l-1)n_P(l) : amp \n" \
           "Igist = Isyn_G_P(l) : amp \n" \
           "Isyn_G_P(l) : amp"

    model_dict['P' + str(nlpc)] = base_model_eq + dd_p.replace('(l)', str(nlpc))
    model_dict['P' + str(nlpc)] = model_dict['P' + str(nlpc)].replace('(l-1)', str(nlpc - 1))

    model_dict['G'] = base_model_eq + "Isynapse = Isyn_I_G : amp\n" \
                                      "Isyn_I_G : amp"

    return model_dict


def create_neurongroup(model_dict, nlayer_dict):
    """
    :param model_dict: dict
    contains AdEx neuron model equation for each unit/layer in the network
    :param nlayer_dict: dict
    contains the number of neurons in each layer/unit of the network
    :return ng_dict: dict
    contains neurons in each layer/unit of the network
    (dict keys: I = input,
                Ex = Error neurons in layer x, Px = Prediction neurons in layer x,
                G = Gist neurons)
    """

    # ng_dict = {}
    # ng_dict = {'I': PoissonGroup(nlayer_dict['I'], rates=np.ones(nlayer_dict['I']) * Hz)}
    ng_dict = {'I': NeuronGroup(N=nlayer_dict['I'],
                                model=model_dict['I'],
                                threshold=thres_cond,
                                reset=reset_cond,
                                refractory=t_ref,
                                method=int_method)
               }
    for ng_key, eq in model_dict.items():
        ng_dict[ng_key] = NeuronGroup(N=nlayer_dict[ng_key],
                                      model=eq,
                                      threshold=thres_cond,
                                      reset=reset_cond,
                                      refractory=t_ref,
                                      method=int_method)
        ng_dict[ng_key].v = EL

    return ng_dict


# def create_syn_models(base_eq, nlayerPC):#, syn_summed_w):
#     '''
#     :param base_eq: str
#     is a base template for a synapse model
#     :param nlayerPC: int
#     is the number of PC layers (one layer = one Error and one Pred)
#     :param syn_summed_w: str
#     is a template for adding a weight variable between two synapses
#     :return syn_dict: dict
#     contains synapse model equations
#     '''
#
#     # syn_ffgn = ['I_E0', 'I_G', 'G_P' + str(nlayerPC)]
#     syn_ffgn = ['I_E0p', 'I_E0n', 'I_G'] + ['G_P' + str(i) for i in range(1,nlayerPC+1)]
#     syn_pc_samelayer = [a.replace('esi', es) for a, es in zip(['E(l)esi_P(l)', 'P(l)_E(l)esi'] * (nlayerPC - 1),
#                                                               np.repeat(['p','n'], (nlayerPC - 1)))]
#     syn_pc_difflayer = [a.replace('esi', es) for a, es in zip(['E(l)esi_P(l+1)', 'P(l+1)_E(l)esi'] * nlayerPC,
#                                                               np.repeat(['p', 'n'], nlayerPC))]
#     synlist = syn_ffgn + syn_pc_samelayer + syn_pc_difflayer
#
#     targets1 = ['(l)']
#     targets2 = ['(l)', '(l+1)']
#
#     syn_dict = {}
#
#     curr_samelayer = 1
#     curr_difflayer = 0
#
#     for i, syn in enumerate(synlist):
#         # syn_ffgn = ['I_E0', 'I_G'] + ['G_P' + str(i) for i in range(1,nlayerPC+1)]
#         if syn in syn_ffgn:
#         # if i < 3:
#             syn_dict[syn] = base_eq.replace('TARGET', syn)
#
#         # syn_pc_samelayer = ['E(l)_P(l)', 'P(l)_E(l)'] * (nlayerPC - 1)
#         # targets1 = ['(l)']
#         elif syn in syn_pc_samelayer:
#         # elif 2 < i < 2 + len(syn_pc_samelayer) + 1:  # (1,1), (2,2)
#             for i1, t1 in enumerate(targets1):
#                 syn = syn.replace(t1, str(curr_samelayer))
#             syn_dict[syn] = base_eq.replace('TARGET', syn)
#             # if i % 2 != 0:
#             if syn[0] == 'E':
#                 syn_dict[syn] += syn_summed_w.replace('TARGET', syn)
#             if syn[0] == 'P':
#                 curr_samelayer += 1
#
#         # syn_pc_difflayer = ['E(l)_P(l+1)', 'P(l+1)_E(l)'] * nlayerPC
#         # targets2 = ['(l)', '(l+1)']
#         elif syn in syn_pc_difflayer:
#         # elif i > 2 + len(syn_pc_samelayer):  # (0,1), (1,2), (2,3)
#             for i2, t2 in enumerate(targets2):
#                 syn = syn.replace(t2, str(curr_difflayer + i2))
#             syn_dict[syn] = base_eq.replace('TARGET', syn)
#             if syn[0] == 'E':
#                 syn_dict[syn] += syn_summed_w.replace('TARGET', syn)
#             if syn[0] == 'P':
#                 curr_difflayer += 1
#
#     return syn_dict

def create_syn_models(ngdict, base_eq, layer_i):
    # es = ['p', 'n']
    syn_dict = {}
    for ng in ngdict.keys():
        if 'I' in ng:
            for esi in es:
                curr_syngroup = ng + '_E0' + esi
                syn_dict[curr_syngroup] = base_eq.replace('TARGET', curr_syngroup)
        syn_dict['I_G'] = base_eq.replace('TARGET', 'I_G')
        if 'G' in ng:
            for ilayer in range(1, layer_i + 1):
                curr_syngroup = ng + '_' + 'P' + str(ilayer)
                syn_dict[curr_syngroup] = base_eq.replace('TARGET', curr_syngroup)
        if 'E' in ng:
            currlayer = int(re.findall(r'\d+', ng)[0])
            # bottom-up pe
            pe_bu = ng + '_' + 'P' + str(currlayer + 1)
            syn_dict[pe_bu] = base_eq.replace('TARGET', pe_bu)
            if currlayer > 0:
                # top-down error
                td_err = ng + '_' + 'P' + str(currlayer)
                syn_dict[td_err] = base_eq.replace('TARGET', td_err)
        if 'P' in ng:
            currlayer = int(re.findall(r'\d+', ng)[0])
            for esi in es:
                if currlayer < layer_i:
                    # input P(l)E(l)
                    inp = ng + '_E' + str(currlayer) + esi
                    syn_dict[inp] = base_eq.replace('TARGET', inp)
                # top-down pred P(l)E(l-1)
                pred_td = ng + '_E' + str(currlayer - 1) + esi
                syn_dict[pred_td] = base_eq.replace('TARGET', pred_td)

    return syn_dict


def create_synapses(syn_dict, ng_dict, iw_dict, layer_n):
    """
    :param syn_dict: dict
    contains synapse model equations
    :param ng_dict: dict
    contains AdEx neuron model equation for each unit/layer in the network
    :param iw_dict: dict
    contains initial weights of FFGN
    :param layer_n:
    the number of PC layer(s)
    :return sg_dict: int
    contains synapse objects
    """

    # Synapses(pre_neuron, post_neuron, model=synapse_model, on_pre=pre_eq, method=syn_cal_method)
    sg_dict = {}

    for syn_key, eq in syn_dict.items():
        pre, post = syn_key.split('_')
        lns = re.findall(r'\d+', syn_key)
        sg_dict[syn_key] = Synapses(ng_dict[pre], ng_dict[post], model=eq, on_pre=pre_eq, method=syn_cal_method)
        print(f'connected {pre} and {post}')
        if (len(lns) > 1) and (lns[0] == lns[1]):
            sg_dict[syn_key].connect('i==j')
            sg_dict[syn_key].w = Wconst
        elif (pre == 'I') and ((post == 'E0p') or (post == 'E0n')):
            sg_dict[syn_key].connect('i==j')
            sg_dict[syn_key].w = Wconst
        else:
            sg_dict[syn_key].connect()

    sg_dict['I_G'].w = iw_dict['I_G']
    for pc_l in range(1, layer_n + 1):
        curr_pred_layer = 'G_P' + str(pc_l)
        sg_dict[curr_pred_layer].w = iw_dict[curr_pred_layer]

    return sg_dict


def create_dws(nlayer_dict, Lw_max):
    """
    :param nlayer_dict: dict
    :param Lw_max: list
    :return ipc_layer_dict: dict
    keys : 'dws' contains weight changes after the last sample in each pc layer.
           'ws'  contains weights in each pc layer.
           The first row contains initial weights. The second row contains updated weights after each sample.
           'sse' contains sum of squared errors in each pc layer.
    """

    # pc_layer_dict = dict.fromkeys(['dws', 'ws', 'sse', 'rep'])
    dict_list = ['dws', 'ws', 'sse', 'rep']
    pc_layer_dict = {}
    for dic in dict_list:
        pc_layer_dict[dic] = {}
        for i in range(nlayerPC):
            if i == 0:
                n_pre, n_post = [nlayer_dict['I'], nlayer_dict['P' + str(i + 1)]]
            else:
                n_pre, n_post = [nlayer_dict['P' + str(i)], nlayer_dict['P' + str(i + 1)]]

            pci = 'PC' + str(i + 1)

            if dic == 'dws':
                pc_layer_dict[dic][pci] = np.zeros((n_pre, n_post))
            elif dic == 'ws':
                pc_layer_dict[dic][pci] = np.zeros((2, n_pre, n_post))
                # np.random.randint(0, wmax, (nlayer_dict['E' + str(0) + 'p'], nlayer_dict['P' + str(1)])).astype(float)
                pc_layer_dict[dic][pci][0] = np.random.randint(0, Lw_max[i], (n_pre, n_post)).astype(float)
                pc_layer_dict[dic][pci][1] = np.copy(pc_layer_dict['ws'][pci][0])
            elif dic == 'sse':
                pc_layer_dict[dic][pci] = np.zeros((nSample*nDigit, niter))
            elif (i < nlayerPC) and (dic == 'rep'):
                pc_layer_dict[dic][pci] = np.zeros((int(niter / nPR), nSample * nDigit,
                                                    nlayer_dict['E' + str(i) + 'p']))

    prmat = {}
    for ipr in range(nlayerPC):
        prmat['PC' + str(ipr + 1)] = np.zeros((nSample * nDigit, nlayer_dict['P' + str(ipr + 1)]))
    gmat = np.zeros((nSample * nDigit, nlayer_dict['G']))

    return prmat, gmat, pc_layer_dict


def lrate_dict(learning_rates, decays, n_iter, n_layerPC):
    lr_dict = {}
    for i_layer in range(1, n_layerPC + 1):
        lr_dict['PC' + str(i_layer)] = [
            np.clip(learning_rates[i_layer - 1] / (1 + np.exp(iter_i * decays[i_layer - 1])), 0.001, 1)
            for iter_i in range(-n_iter, n_iter, 2)]

    return lr_dict


def create_monitors(ng_dict, iteri, ngdict):
    # set up monitors for PC2 + FFN
    err_vars = ['x_trace', 'Ierror', 'Iinput', 'Ipred']
    pred_vars = ['x_trace', 'Igist', 'Isynapse']  # , 'Ierror_bu'] #'Ierror_td'

    monitor_dict = {}
    for ng_key, ng in ng_dict.items():
        if 'E' in ng_key:
            monitor_dict[ng_key] = StateMonitor(ng, err_vars, record=True)
        elif 'P' in ng_key:
            monitor_dict[ng_key] = StateMonitor(ng, pred_vars, record=True)
            # if (iteri+1) >= 10:
            #     if ng_key == 'P1':
            #         curr_layer = int(re.findall(r'\d+', ng_key)[0])
            #         pred_vars_plus = [pv for pv in pred_vars]
            #         for esi in es:
            #             pred_vars_plus.append('Isyn_E' + str(curr_layer-1) + esi + '_P' + str(curr_layer))
            #             pred_vars_plus.append('Isyn_E' + str(curr_layer) + esi + '_P' + str(curr_layer))
            #         monitor_dict[ng_key] = StateMonitor(ng, pred_vars_plus, record=True)
            #     else:
            #         monitor_dict[ng_key] = StateMonitor(ng, pred_vars, record=True)
            # else:
            #     monitor_dict[ng_key] = StateMonitor(ng, pred_vars, record=True)
            # if str(nlayerPC) in key:
            #     monitor_dict[key] = StateMonitor(ng, state_vars[:2] + state_vars[-2:], record=True)
            # else:
            #     monitor_dict[key] = StateMonitor(ng, state_vars[:2] + state_vars[-3:], record=True)
        # else:
        elif 'G' in ng_key:
            monitor_dict[ng_key] = StateMonitor(ng, 'Isynapse', record=True)

    return monitor_dict


def weight_clipping(weight, error_signal, pred_signal, max_dw, max_w, lr):
    dw = lr * (np.mean(error_signal, axis=2) * np.mean(pred_signal, axis=2).T)
    dw = np.clip(dw, -max_dw, max_dw)
    weight += dw  # + 2 * 0.001 * We1p1
    weight = np.clip(weight, 0, max_w)

    return dw, weight


def init_weights(nLayerDict, nlayerPCs, ffw_maxlist, eta):
    # INITIALIZE WEIGHTS
    conn_p_ig = eta * (nLayerDict['I'] + nLayerDict['G']) / (nLayerDict['I'] * nLayerDict['G'])

    initw_dict = {'I_G': np.random.normal(ffw_maxlist[0],
                                          ffw_maxlist[0] / 10,
                                          (nLayerDict['G'] * nLayerDict['I'])) *
                         (np.random.random((nLayerDict['G'] * nLayerDict['I'])) < conn_p_ig)}

    for ilayer in range(1, nlayerPCs + 1):
        curr_pl = 'P' + str(ilayer)
        gp_w = ffw_maxlist[ilayer]
        curr_conn_p = eta * (nLayerDict['G'] + nLayerDict[curr_pl]) / (nLayerDict['G'] * nLayerDict[curr_pl])
        initw_dict['G_' + curr_pl] = np.random.normal(gp_w, gp_w / 10, (nLayerDict[curr_pl] * nLayerDict['G'])) * \
                                     (np.random.random((nLayerDict[curr_pl] * nLayerDict['G'])) < curr_conn_p)

    return initw_dict


def reordering(order_mat, idx):
    len_mat = np.arange(len(idx))
    reorder = np.ravel([np.where(idx == i)[0] for i in len_mat])
    new_mat = order_mat[reorder]

    return new_mat


def plot_testset(testset, testset_idx):  # , digits):

    testset_len, testset_x, testset_y = testset.shape
    # X = reordering(testset.reshape(nDigit * nSample, 784), testset_idx)
    raw_img = reordering(testset.reshape(testset_len, testset_x * testset_y), testset_idx)
    # Plot images of the digits
    fig = plt.figure()

    img = np.zeros(((testset_x + 2) * nDigit, (testset_y + 2) * nSample))
    for i in range(nDigit):
        ix = (testset_x + 2) * i + 1
        for j in range(nSample):  # n_img_per_row):
            iy = (testset_y + 2) * j + 1
            img[ix:ix + testset_x, iy:iy + testset_y] = raw_img[i * nSample + j].reshape((testset_x, testset_y))

    plt.imshow(img, cmap="Reds")
    plt.xticks([])
    plt.yticks([])
    plt.title('MNIST images')
    plt.savefig('figures/normalized_digits')

    plt.show()

    return fig


def plot_tsne(img, img_dict):
    testset_len, testset_x, testset_y = img.shape
    img_mat = img.reshape(testset_len, testset_x * testset_y)
    # test_set_dict = np.load(img_dict)

    digits = img_dict['digits']
    digits.sort()
    test_set_idx = img_dict['test_set_idx']
    img_reordered = reordering(img_mat, test_set_idx)
    y_train = np.repeat(digits, 5)

    x_all = pd.DataFrame(img_reordered)
    y_all = pd.DataFrame(y_train)

    plot_x = x_all.sample(frac=1, random_state=10).reset_index(drop=True)

    plot_y = y_all.sample(frac=1, random_state=10).reset_index(drop=True)

    tsne = TSNE()
    tsne_results = tsne.fit_transform(plot_x.values)

    plot_x['label'] = plot_y

    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, title='TSNE')
    # Create the scatter
    scatter_plot = ax.scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        c=plot_x['label'],
        cmap=plt.cm.get_cmap('Paired'),
        alpha=1)
    plot_legend = ax.legend(*scatter_plot.legend_elements(), loc='lower left', title='digits')
    ax.add_artist(plot_legend)
    # ax.legend()

    plt.savefig('figures/tsne_plot')
    plt.show()

    return fig


def matrix_rdm(matrix_data):
    output = np.array([1 - stats.spearmanr(matrix_data[ni].flatten(), matrix_data[mi].flatten())[0]
                       #                    1 - stats.pearsonr(matrix[ni].flatten(), matrix[mi].flatten())[0]
                       for ni in range(len(matrix_data))
                       for mi in range(len(matrix_data))]).reshape(len(matrix_data), len(matrix_data))

    return output


def rsa_analysis2(mats, rdm_titles, dataset_idx, digits, figname):
    digits.sort()

    fig, rsa_axes = plt.subplots(ncols=len(mats) + 1, nrows=1,
                                 sharex=True, sharey=True, figsize=(5 * (len(mats) + 1), 5))
    rsa_axes = rsa_axes.flatten()
    # if len(mats) > 1:
    #     axes = axes.flatten()
    # else:
    #     axes = [axes]

    for i in range(len(rsa_axes)):
        if i < len(rsa_axes) - 1:
            reordered = reordering(mats[i], dataset_idx)
            rdm = matrix_rdm(reordered)
            rsa_axes[i].imshow(rdm, cmap='Reds', vmin=0, vmax=1, aspect='auto')

            rsa_axes[i].set_title(rdm_titles[i])
            rsa_axes[i].set(xlabel='digit #', ylabel='digit#')
            rsa_axes[i].set_xticks(np.arange(nDigit * nSample))
            rsa_axes[i].set_xticklabels(np.repeat(digits, nSample))  # , fontsize=6)
            rsa_axes[i].set_yticks(np.arange(nDigit * nSample))
            rsa_axes[i].set_yticklabels(np.repeat(digits, nSample))  # , fontsize=6)
            rsa_axes[i].label_outer()
        else:
            tmat = np.ones((np.shape(mats[0])[0], np.shape(mats[0])[0]))
            for ii in range(nDigit):
                tmat[nSample * ii:nSample * ii + nSample,
                nSample * ii:nSample * ii + nSample] = 0
            rdmplot = rsa_axes[i].imshow(tmat, cmap='Reds', vmin=0, vmax=1, aspect='auto')

            rsa_axes[i].set_title('ideal classifier')
            rsa_axes[i].set(xlabel='digit #', ylabel='digit#')
            rsa_axes[i].set_xticks(np.arange(nDigit * nSample))
            rsa_axes[i].set_xticklabels(np.repeat(digits, nSample))  # , fontsize=6)
            rsa_axes[i].set_yticks(np.arange(nDigit * nSample))
            rsa_axes[i].set_yticklabels(np.repeat(digits, nSample))  # , fontsize=6)
            rsa_axes[i].label_outer()

    fig.colorbar(rdmplot, ax=rsa_axes[-1], label='dissimilarity (1-corr)', shrink=0.6)
    fig.suptitle('representational dissimilairty matrix')
    fig.tight_layout()

    plt.savefig(figname)

    return fig


def create_bgs(nlayer_dict):
    bgs = {}
    for layer, nlayer in nlayer_dict.items():
        if layer in ['I', 'G']:
            bgs[layer[:2]] = 0
        else:
            if layer[:2] not in bgs:
                bgs[layer[:2]] = np.random.normal(500, 250, nlayer)

    return bgs


def plot_pc1rep(rep_dict, testsetidx):
    testset_len, testset_x, testset_y = [np.shape(rep_dict)[0],
                                         int(np.sqrt(np.shape(rep_dict)[1])),
                                         int(np.sqrt(np.shape(rep_dict)[1]))]
    ordered_img = reordering(rep_dict, testsetidx)
    # img_xy = int(np.sqrt(nI))

    fig = plt.figure()
    img = np.zeros(((testset_x + 2) * nDigit, (testset_y + 2) * nSample))
    for i in range(nDigit):
        ix = (testset_x + 2) * i + 1
        for j in range(nSample):
            iy = (testset_y + 2) * j + 1
            img[ix:ix + testset_x, iy:iy + testset_y] = ordered_img[i * nSample + j].reshape((testset_x, testset_y))

    plt.imshow(img, cmap='Reds', vmin=0, vmax=3000)
    plt.xticks([])
    plt.yticks([])
    plt.title('L1 representations of MNIST images')
    # plt.savefig('figures/L1rep')
    # plt.show()

    return fig


def plot_progress(nlayerpc, nlayerdict, monitordict, pclayerdict, iteri, imagei, nimages):

    attr_list = ['Iinput', 'Ipred', 'Ierror']
    col_size = 4
    row_size = nlayerpc + 1
    fig, axes_analysis = plt.subplots(ncols=col_size, nrows=row_size, figsize=(5 * col_size, 5 * row_size))
    # [:,0] Input signals at E+
    # [:,1] Pred signals at E+
    # [:,2] Error signals at E+
    for rowi in range(nlayerpc):
        curr_err = 'E(l)p'.replace('(l)', str(rowi))
        for attr, colj in zip(attr_list, range(len(attr_list))):
            if attr == 'Ierror':
                cmaps = 'bwr'
                vmin = -2000
                vmax = 2000
            else:
                cmaps = 'Reds'
                vmin = None  # 0
                vmax = None  # 3000
            imshow_plot = axes_analysis[rowi, colj].imshow(
                (monitordict[curr_err].__getattr__(attr)[:, -last_t:] / pamp).mean(axis=1).reshape(
                    int(np.sqrt(nlayerdict[curr_err])),
                    int(np.sqrt(nlayerdict[curr_err]))),
                cmap=cmaps, vmin=vmin, vmax=vmax, aspect='auto')
            fig.colorbar(imshow_plot, ax=axes_analysis[rowi, colj], shrink=0.6, label='current (pamp)')
            axes_analysis[rowi, colj].set_xticks([])
            axes_analysis[rowi, colj].set_yticks([])
            axes_analysis[rowi, colj].set_title('L' + str(rowi + 1) + ' ' + attr)
        # [:,3] SSE
        if iteri < 3:
            sse_idx = 1
        else:
            sse_idx = 3
        # axes_analysis[rowi, 3].plot(pclayerdict['sse']['PC' + str(rowi + 1)][sse_idx:iteri * nimages + imagei + 1])
        axes_analysis[rowi, 3].plot(np.arange(sse_idx, iteri+1), pclayerdict['sse']['PC' + str(rowi + 1)][imagei, sse_idx : iteri + 1])
        axes_analysis[rowi, 3].set_title('L' + str(rowi + 1) + ' SSE : image #{0} iter# {1}'.format(imagei, iteri+1))
    #     pc_layer_dict[dic][pci] = np.zeros((nSample*nDigit, niter))

    # [row_size, 0] input
    # inp_plot = axes_analysis[nlayerpc, 0].imshow(
    #     (monitordict['I'].Isynapse[:, -last_t:] / pamp).mean(axis=1).reshape(
    #         int(np.sqrt(nlayerdict['I'])),
    #         int(np.sqrt(nlayerdict['I']))),
    #     cmap='Reds', aspect='auto')
    # fig.colorbar(inp_plot, ax=axes_analysis[nlayerpc, 0], shrink=0.6, label='current (pamp)')
    # axes_analysis[nlayerpc, 0].set_xticks([])
    # axes_analysis[nlayerpc, 0].set_yticks([])
    # axes_analysis[nlayerpc, 0].set_title('Input')
    # [row_size, 1]
    gsyn_plot = axes_analysis[nlayerpc, 1].imshow(
        (monitordict['G'].Isynapse[:, -last_t:] / pamp).mean(axis=1).reshape(
            int(np.sqrt(nlayerdict['G'])),
            int(np.sqrt(nlayerdict['G']))),
        cmap='Reds', aspect='auto')
    fig.colorbar(gsyn_plot, ax=axes_analysis[nlayerpc, 1], shrink=0.6, label='current (pamp)')
    axes_analysis[nlayerpc, 1].set_xticks([])
    axes_analysis[nlayerpc, 1].set_yticks([])
    axes_analysis[nlayerpc, 1].set_title('gist at G')
    # [row_size, 2]
    gp1_plot = axes_analysis[nlayerpc, 2].imshow(
        (monitordict['P1'].Igist[:, -last_t:] / pamp).mean(axis=1).reshape(
            int(np.sqrt(nlayerdict['P1'])),
            int(np.sqrt(nlayerdict['P1']))),
        cmap='Reds', aspect='auto')
    fig.colorbar(gp1_plot, ax=axes_analysis[nlayerpc, 2], shrink=0.6, label='current (pamp)')
    axes_analysis[nlayerpc, 2].set_xticks([])
    axes_analysis[nlayerpc, 2].set_yticks([])
    axes_analysis[nlayerpc, 2].set_title('gist at P1')
    # [row_size, 3]
    if nlayerpc > 1:
        gp2_plot = axes_analysis[nlayerpc, 3].imshow(
            (monitordict['P2'].Igist[:, -last_t:] / pamp).mean(axis=1).reshape(
                int(np.sqrt(nlayerdict['P2'])),
                int(np.sqrt(nlayerdict['P2']))),
            cmap='Reds', aspect='auto')
        fig.colorbar(gp2_plot, ax=axes_analysis[nlayerpc, 3], shrink=0.6, label='current (pamp)')
        axes_analysis[nlayerpc, 3].set_xticks([])
        axes_analysis[nlayerpc, 3].set_yticks([])
        axes_analysis[nlayerpc, 3].set_title('gist at P2')

    figname = 'iter{}_training_digit{}'.format(iteri + 1, imagei)

    # end_plot_time = time.process_time()
    plt.savefig('figures/' + figname)
    plt.show()

    return fig


def rdm2_plot(matlist, mat_idx, digit_list, label_rdms):
    rematlist = [reordering(mat_i, mat_idx) for mat_i in matlist]
    rdmlist = [matrix_rdm(remat) for remat in rematlist]

    tmat = np.ones((np.shape(matlist[0])[0], np.shape(matlist[0])[0]))
    for i in range(len(digit_list)):
        tmat[i * nSample : i * nSample + nSample, i * nSample : i * nSample + nSample] = 0
    rdmlist.append(tmat)

    digit_list.sort()

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(nrows=2, ncols=len(rdmlist), wspace=0.1)
    for i in range(len(rdmlist)):
        rdm_plot = fig.add_subplot(gs[0, i])
        rdm_plot.imshow(rdmlist[i], cmap='Reds', aspect='auto', vmin=0, vmax=1)
        rdm_plot.set_title(label_rdms[i])
        rdm_plot.set(xlabel='digit #', ylabel='digit#')
        rdm_plot.set_xticks(np.arange(np.shape(rdmlist[i])[0]))
        rdm_plot.set_xticklabels(np.repeat(digit_list, nSample))  # , fontsize=6)
        rdm_plot.set_yticks(np.arange(np.shape(rdmlist[i])[0]))
        rdm_plot.set_yticklabels(np.repeat(digit_list, nSample))  # , fontsize=6)
        rdm_plot.label_outer()
        # if i == len(matlist)-1:
        #     ax = plt.subplot()
        #     plt.colorbar(rdm_plot, cax=gs[0,i], label='dissimilarity (1-corr)', shrink=0.6)

    rdm_triu = [np.triu(rdm) for rdm in rdmlist]
    r2_is = [1 - stats.spearmanr(rdm_triu[0].flatten(), rdm_triu[i].flatten())[0] for i in range(1, len(rdmlist) - 1)]
    # r2I_tuples = [(0,1), ... , (0,n-1)]
    r2_ts = [1 - stats.spearmanr(rdm_triu[-1].flatten(), rdm_triu[i].flatten())[0] for i in range(1, len(rdmlist) - 1)]
    # r2T_tuples = [(n,1), ... , (n,n-1)]

    r2_i = fig.add_subplot(gs[1, :2])
    r2_i.bar(np.arange(len(r2_is)), r2_is)
    r2_i.set_xticks(np.arange(len(r2_is)))
    r2_i.set_xticklabels([label_rdms[0] + ' - ' + i for i in label_rdms[1:-1]])
    r2_i.set_ylim([0, 1])
    r2_i.set_ylabel('1-Spearman corr')
    r2_i.set_title('deviation from input')

    r2_t = fig.add_subplot(gs[1, 2:])
    r2_t.bar(np.arange(len(r2_ts)), r2_ts)
    r2_t.set_xticks(np.arange(len(r2_ts)))
    r2_t.set_xticklabels([label_rdms[-1] + ' - ' + i for i in label_rdms[1:-1]])
    r2_t.set_yticks([])
    r2_t.set_ylim([0, 1])
    r2_t.set_title('deviation from an ideal classifier')

    return fig


def plot_gist(n_pc_layer, monitordict, nlayerdict):
    row_size = n_pc_layer + 1
    fig, axes_analysis = plt.subplots(ncols=1, nrows=row_size, figsize=(5, 5 * row_size))

    for layer_i in range(n_pc_layer + 1):
        if layer_i == 0:
            # [row_size, 1]
            gsyn_plot = axes_analysis[layer_i].imshow(
                (monitordict['G'].Isynapse[:, -last_t:] / pamp).mean(axis=1).reshape(
                    int(np.sqrt(nlayerdict['G'])),
                    int(np.sqrt(nlayerdict['G']))),
                cmap='Reds', aspect='auto')
            fig.colorbar(gsyn_plot, ax=axes_analysis[layer_i], shrink=0.6, label='current (pamp)')
            axes_analysis[layer_i].set_title('gist at G')
        else:
            curr_pl = 'P{0}'.format(layer_i)
            gpx_plot = axes_analysis[layer_i].imshow(
                (monitordict[curr_pl].Igist[:, -last_t:] / pamp).mean(axis=1).reshape(
                    int(np.sqrt(nlayerdict[curr_pl])),
                    int(np.sqrt(nlayerdict[curr_pl]))),
                cmap='Reds', aspect='auto')
            fig.colorbar(gpx_plot, ax=axes_analysis[layer_i], shrink=0.6, label='current (pamp)')
            axes_analysis[layer_i].set_title('gist at ' + curr_pl)

        axes_analysis[layer_i].set_xticks([])
        axes_analysis[layer_i].set_yticks([])

    return fig
