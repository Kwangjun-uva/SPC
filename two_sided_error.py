import numpy as np
import matplotlib.pyplot as plt
import winsound
import pickle
# from brian2 import *
from network_functions_2ER import *

## create n x n sample grid images
# sample_size = 5
# sample_grid = np.zeros((nSample*nDigit, sample_size, sample_size))
# for isample in range(nSample):
#     for idigit in range(nDigit):
#         if nDigit*isample+idigit < 2:
#             sample_grid[nDigit*isample+idigit] = np.zeros((sample_size, sample_size))
#             sample_grid[nDigit*isample+idigit, 1:4, 1:4] = np.random.normal(0.7, 0.2, (3,3))
#             sample_grid[nDigit*isample+idigit, 2, 2] = 0
#         else:
#             sample_grid[nDigit * isample + idigit] = np.zeros((sample_size, sample_size))
#             for xi in range(1,4):
#                 sample_grid[nDigit*isample+idigit, xi, xi] = np.random.normal(0.7, 0.2)
#             sample_grid[nDigit*isample+idigit, 3, 1] = np.random.normal(0.7, 0.2)
#             sample_grid[nDigit*isample+idigit, 1, 3] = np.random.normal(0.7, 0.2)
# imgs = sample_grid.reshape(1,sample_size, sample_size)
# img = np.load('figures/testset_data.npy').reshape(sample_size,sample_size)
# fig, ax = plt.subplots()
# img_plot = ax.matshow(imgs[0], cmap='Reds')
# fig.colorbar(img_plot, ax=ax, shrink=0.6)
# for (i, j), z in np.ndenumerate(imgs[0]):
#     ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
# imgs = sample_grid[test_set_idx]

# load MNIST images
imgs = np.load('figures/testset_data.npy')
test_set_dict = np.load('figures/testset_dict.npz')

digits = test_set_dict['digits']
test_set_idx = test_set_dict['test_set_idx']
sample_set_idx = test_set_dict['sample_set_idx']

nimages = len(imgs)

# # load top-down feeds
# with open('figures/20210217/55_4layer/all_rep.pkl', 'rb') as handle2:
#     top_down_feed = pickle.load(handle2)
#
# digits = [4, 0]
# test_set_idx = [0, 1]
# sample_set_idx = [0, 1]
#
# randomDigit_idx = [5, 17]
# td_feed_img_dict = {}
#
# fig_tdfeed, axs = plt.subplots(nrows=4, ncols=len(randomDigit_idx), figsize=(20,10))
# for iCol, iDigit in enumerate(randomDigit_idx):
#     td_feed_img_dict[iCol] = {}
#     for iLayer in range(4):
#         curr_layer = 'PC' + str(iLayer + 1)
#         reshape_size = np.sqrt(top_down_feed[curr_layer][-1, 0].shape[0]).astype(int)
#
#         td_feed_img_dict[iCol][curr_layer] = top_down_feed[curr_layer][-1, iDigit].reshape(reshape_size, reshape_size)
#
#         axs[iLayer, iCol].imshow(top_down_feed[curr_layer][-1, iDigit].reshape(reshape_size, reshape_size), cmap=plt.cm.Reds)
#         axs[iLayer, iCol].set_xticks([])
#         axs[iLayer, iCol].set_yticks([])
#         axs[iLayer, iCol].set_title(curr_layer + 'digit #{0}'.format(iCol+1))
#
# fig_tdfeed.tight_layout()
# plt.show()
# nimages = len(td_feed_img_dict.keys())

# plot images (mnist)
img_plot = plot_testset(imgs, test_set_idx)
plt.show()

# # plot tsne
# tsne_plot = plot_tsne(imgs, test_set_dict)
# plt.show()

# LEARNING RATES
lrate_list = [0.0] * nlayerPC  # for each PC layer (length = nlayerPC)
decay_list = [0] * nlayerPC  # for each PC layer (length = nlayerPC)
lr_dict = lrate_dict(lrate_list, decay_list, niter, nlayerPC)

# SPECIFY THE NETWORK STRUCTURE
# sample_size = np.shape(imgs)[1]
sample_size = 28
nlayer_dict = network_size(nlayerlist=[sample_size ** 2, 30 ** 2, 25 ** 2, 20 ** 2, 15**2, 10 ** 2], nlpc=nlayerPC)

# CREATE MEMORY FOR GIST, WEIGHT, WEIGHT UPDATE
Lw_max = [50] * nlayerPC # length : nlayerPC
prMat, gistMat, pc_layer_dict = create_dws(nlayer_dict, Lw_max)
# assign pre-trained weights
with open('figures/20210217/55_4layer/ws_dict_100.pkl', 'rb') as pc_ws:
    pc_weights = pickle.load(pc_ws)
for pc_key, ws_pci in pc_weights.items():
    pc_layer_dict['ws'][pc_key][1] = ws_pci[1]

# INITIALIZE WEIGHTS
ffw = [250] + [300] * nlayerPC  # igWmax, gp1Wmax, ..., gpnWmax
eta = 0.25
initw_dict = init_weights(nlayer_dict, nlayerPC, ffw, eta)

# assign the same gist weights used during the training
with open('figures/20210217/55_4layer/gist_weights.pkl', 'rb') as gist_ws:
   gist_weights = pickle.load(gist_ws)
for gist_key, gist_pci in gist_weights.items():
    initw_dict[gist_key] = gist_pci

# # clear cache for cython
# try:
#     clear_cache('cython')
#     print('cache cleared')
# except Exception:
#     pass

start_scope()
defaultclock.dt = 1 * ms

# CREATE NEURON GROUPS
ng_model_dict = create_neuron_models(nlayerPC, basic_eq, template_Pred_model, template_Err_model)
ng_dict = create_neurongroup(ng_model_dict, nlayer_dict)
# CREATE SYNAPSES
syn_model_dict = create_syn_models(ngdict=ng_dict, base_eq=eq_syn, layer_i=nlayerPC)
sg_dict = create_synapses(syn_model_dict, ng_dict, initw_dict, nlayerPC)

# # feed neural representations learned from training sessions
# td_feed_layer = 3
#
# td_feed_eq = """
# dv/dt = (DeltaT*gL*exp((-VT + v)/DeltaT) + Isynapse - c + gL*(EL - v))/Cm : volt
# dc/dt = (a*(-EL + v) - c)/tauw : amp
#
# dx_trace/dt = -x_trace/tau_s - x_up/tau_rise : radian
# dx_up/dt = -x_up/tau_rise : radian
#
# Isynapse : amp
# Igist = Isyn_G_P3 : amp
#
# Isyn_E2n_P3 : amp
# Isyn_E2p_P3 : amp
# Isyn_E3n_P3 : amp
# Isyn_E3p_P3 : amp
# Isyn_G_P3 : amp
# """
# ng_dict['P' + str(td_feed_layer)]= NeuronGroup(N=nlayer_dict['P' + str(td_feed_layer)],
#                                                model=td_feed_eq,
#                                                threshold=thres_cond,
#                                                reset=reset_cond,
#                                                refractory=t_ref,
#                                                method=int_method)

# store network
net = Network(collect())
net.add(ng_dict)
net.add(sg_dict)

net.store()

repMat_idx = 0
repMat_count = 0

start_total_time = time.process_time()
for iter_i in range(niter):
    start_iter = time.process_time()

    if repMat_count == nimages:
        repMat_idx += 1
        repMat_count = 0

    for image_i in range(nimages):
        # load the network
        net.restore()

        # # # simulation
        # add monitors
        monitor_dict = create_monitors(ng_dict, iter_i, ng_dict)
        net.add(monitor_dict)

        # forward sweep from input to gist to Pred layers ('gist')
        # feed in the input image
        ng_dict['I'].Iext_mat = (12000 * imgs[image_i].flatten() + 600) * pamp
            # ng_dict['I'].Iext = imgs[image_i].flatten() * pamp  # (2400 * sample_grid.flatten() + 600) * pamp
        # ng_dict['I'].rates = imgs[image_i].flatten() * kHz

        # # top-down feed from layer X
        # ng_dict['P' + str(td_feed_layer)].Isynapse = td_feed_img_dict[image_i]['PC' + str(td_feed_layer)].flatten() * pamp
#
        # simulate FFN
        start_ffgn_run = time.time()
        net.run(runtime_ff * ms)
        end_ffgn_run = time.time()

        # save gist and prior representations
        if iter_i == 0:
            gistMat[image_i] = (monitor_dict['G'].Isynapse[:, :runtime_ff] / pamp).mean(axis=1)
            for iprl in range(nlayerPC):
                prMat['PC' + str(iprl + 1)][image_i] = (
                        monitor_dict['P' + str(iprl + 1)].Igist[:, :runtime_ff] / pamp).mean(axis=1)

        # gist_plots = plot_gist(n_pc_layer=nlayerPC,
        #                        monitordict=monitor_dict,
        #                        nlayerdict=nlayer_dict)
        # plt.show()

        # remove monitors
        for dict_key, val in monitor_dict.items():
            net.remove(val)
        # net.remove(monitor_dict)
        monitor_dict.clear()
        # add monitors
        monitor_dict = create_monitors(ng_dict, iter_i, ng_dict)
        net.add(monitor_dict)

        # # pc network to infer causes
        # initialize pc synaptic weights
        for i in range(nlayerPC):
            for signs in ['p', 'n']:
                sg_dict['E' + str(i) + signs + '_P' + str(i + 1)].w = pc_layer_dict['ws']['PC' + str(i + 1)][
                    1].flatten()
                sg_dict['P' + str(i + 1) + '_E' + str(i) + signs].w = pc_layer_dict['ws']['PC' + str(i + 1)][
                    1].T.flatten()

        # simulate pc_ffn
        start_pcffgn_run = time.time()
        for key, val in monitor_dict.items():
            val.active = False
        net.run((runtime_pc - last_t - 1) * ms)
        for key, val in monitor_dict.items():
            val.active = True
        net.run((last_t + 1) * ms)
        end_pcffgn_run = time.time()

        print(f'iter{iter_i + 1}/{niter}, img{image_i + 1}/{nimages} '
              f'done in {end_pcffgn_run - start_ffgn_run:.2f} sec: '
              f'ffgn = {end_ffgn_run - start_ffgn_run:.2f} sec, pc = {end_pcffgn_run - start_pcffgn_run:.2f} sec')
        time_left = (end_pcffgn_run - start_ffgn_run) * ((niter * nimages) - (iter_i * nimages + (image_i + 1)))
        print(f'estimated time of completion: {datetime.timedelta(seconds=time_left)}')

        # # WEIGHT UPDATES
        # hebb_dict = {}
        #
        # for i in range(nlayerPC):
        #     pci = 'PC' + str(i + 1)
        #
        #     # store mean errors
        #     Ierror_dict = np.mean(np.abs(monitor_dict['E' + str(i) + 'p'].Ierror[:, -last_t:] / pamp), axis=1)
        #     # store SSE
        #     pc_layer_dict['sse'][pci][image_i, iter_i] = np.sum(Ierror_dict ** 2)
        #     # pc_layer_dict['sse'][pci][iter_i * nimages + image_i + 1] = np.sum(
        #     #     Ierror_dict[pci] ** 2)
        #
        #     hebb_dict[pci] = {}
        #     # postsynaptic variable :
        #     # hebb_dict[pci]['post'] = np.einsum('ij,ik->ijk',
        #     #                                    pc_layer_dict['ws'][pci][1].T,
        #     #                                    monitor_dict['P' + str(i + 1)].x_trace[:, -last_t:])
        #     hebb_dict[pci]['post'] = monitor_dict['P' + str(i + 1)].x_trace[:, -last_t:]
        #
        #     # # resilience to learning hebb_dict[pci]['res'] = np.tile(Ierror_dict[pci] > dw_lim, nlayer_dict['P' +
        #     # str(i + 1)]).reshape(nlayer_dict['P' + str(i + 1)], nlayer_dict['E' + str(i) + 'p']).T * 1 L1
        #     # regularization error
        #     # hebb_dict[pci]['L1reg'] = (Lw_max[i] / l1reg_alpha) * np.sign(pc_layer_dict['ws'][pci][1])
        #     hebb_dict[pci]['L2reg'] = l2reg_alpha * 2 * pc_layer_dict['ws'][pci][1]
        #
        #     pc_layer_dict['dws'][pci] = {}
        #     for err_sign in ['p', 'n']:
        #         err_label = 'E' + str(i) + err_sign
        #         # presynaptic variable :
        #         # hebb_dict[pci]['pre'] = np.einsum('ij,ik->ijk',
        #         #                                   pc_layer_dict['ws'][pci][1],
        #         #                                   monitor_dict[err_label].x_trace[:, -last_t:])
        #         hebb_dict[pci]['pre'] = monitor_dict[err_label].x_trace[:, -last_t:]
        #         # # weight update and clamping
        #         # hebb_val = lr_dict[pci][iter_i] * \
        #         #            hebb_dict[pci]['post'].mean(axis=2).T * \
        #         #            hebb_dict[pci]['pre'].mean(axis=2) * \
        #         #            hebb_dict[pci]['res']
        #         #
        #         # if err_sign == 'p':
        #         #     reg_sign = -1
        #         # else:
        #         #     reg_sign = 1
        #         #
        #         # pc_layer_dict['dws'][pci][err_sign] =(lr_dict[pci][iter_i] / 2) * \
        #         #            (hebb_dict[pci]['post'].mean(axis=2).T * hebb_dict[pci]['pre'].mean(axis=2) +
        #         #             reg_sign * hebb_dict[pci]['L1reg'])
        #
        #         # pc_layer_dict['dws'][pci][err_sign] = np.clip(hebb_val, -Lw_max[i]*0.5, Lw_max[i]*0.5)
        #         # pc_layer_dict['dws'][pci][err_sign] = lr_dict[pci][iter_i] * \
        #         #                                       (hebb_dict[pci]['post'].mean(axis=2).T *
        #         #                                        hebb_dict[pci]['pre'].mean(axis=2) -
        #         #                                        hebb_dict[pci]['L2reg'])
        #         pc_layer_dict['dws'][pci][err_sign] = lr_dict[pci][iter_i] * \
        #                                               (np.einsum('ij,kj->ikj',
        #                                                         hebb_dict[pci]['pre'],
        #                                                         hebb_dict[pci]['post']).mean(axis=2) - \
        #                                               hebb_dict[pci]['L2reg'])
        #
        #     pc_layer_dict['ws'][pci][1] += pc_layer_dict['dws'][pci]['p'] - pc_layer_dict['dws'][pci]['n']
        #     pc_layer_dict['ws'][pci][1] = np.clip(pc_layer_dict['ws'][pci][1], 0, inf)#Lw_max[i])
        #
        # # del hebb_dict

        if ((iter_i + 1) % nPR == 0) or (iter_i == 0):

            for i in range(nlayerPC):
                for err_sign in ['p', 'n']:
                    err_label = 'E' + str(i) + err_sign
                    pci = 'PC' + str(i + 1)
                    pc_layer_dict['rep'][pci][repMat_idx, image_i] = (
                            monitor_dict[err_label].Ipred[:, -last_t:] / pamp).mean(axis=1)

            if (iter_i + 1) % nPR == 0:
                repMat_count += 1

            if image_i in sample_set_idx:
                plot_progress(nlayerpc=nlayerPC,
                              nlayerdict=nlayer_dict,
                              monitordict=monitor_dict,
                              pclayerdict=pc_layer_dict,
                              iteri=iter_i,
                              imagei=image_i,
                              nimages=nimages)

        if iter_i < niter:
            for dict_key, val in monitor_dict.items():
                net.remove(val)
                # net.remove(monitor_dict)
            monitor_dict.clear()

    if iter_i == 0:
        # check on pfc
        ff_rdm_plot = rsa_analysis2([imgs, gistMat] + [prMat['PC' + str(il + 1)] for il in range(nlayerPC)],
                                    ['normalized images', 'gist'] + [f'P{il + 1} gist' for il in range(nlayerPC)],
                                    test_set_idx, digits, 'figures/rdm_gist')
        plt.show()
    #     # for dict_key, val in monitor_dict.items():
    #     #     net.remove(val)
    #     #     # net.remove(monitor_dict)
    #     # monitor_dict.clear()

    end_iter = time.process_time()
    print('\n========iteration {0}/{1} ended in {2:.2f} sec ========'.format(iter_i + 1, niter, end_iter - start_iter))

end_total_time = time.process_time()
print(f'time of completion: {datetime.timedelta(seconds=end_total_time - start_total_time)}')

with open('figures/rep_dict.pkl', 'wb') as handle:
    pickle.dump(pc_layer_dict['rep'], handle)
with open('figures/sse_dict.pkl', 'wb') as handle2:
    pickle.dump(pc_layer_dict['sse'], handle2)
with open('figures/ws_dict.pkl', 'wb') as handle3:
    pickle.dump(pc_layer_dict['ws'], handle3)

for iter_j in range(int(niter / nPR)):
    # plot L1 rep
    L1rep_fig = plot_pc1rep(pc_layer_dict['rep']['PC1'][iter_j], test_set_idx)
    plt.savefig('figures/L1rep_{0}'.format(int(iter_j * nPR + nPR)))
    plt.show()

    # RDM2
    rdm_list = [imgs.reshape(nSample * nDigit, nlayer_dict['I'])]
    [rdm_list.append(pc_layer_dict['rep']['PC' + str(i + 1)][iter_j]) for i in range(nlayerPC)]

    # RDM2 plot
    label_rdms = ['I']
    [label_rdms.append(f'L{i + 1}') for i in range(nlayerPC)]
    label_rdms.append('T')
    rsa_plot = rdm2_plot(rdm_list, test_set_idx, digits, label_rdms)
    plt.savefig('figures/rdms_{0}'.format(int(iter_j * nPR + nPR)))
    plt.show()

winsound.Beep(frequency=940, duration=1000)
#
#
#
#
#
# # #
# # # for i in range(nlayerPC):
# # #     np.save(f'figures/v3_rep{i + 1}Digit',
# # #             pc_layer_dict['rep']['PC' + str(i + 1)].reshape(int(niter / nPR) * len(imgs), nlayer_dict['E' + str(i)]))
# # #
# # #     en_input_sig = (monitor_dict['E0n'].Iinput[:, -last_t:]/pamp).mean(axis=1).reshape(sample_size, sample_size)
# # #     en_pred_sig = (monitor_dict['E0n'].Ipred[:, -last_t:]/pamp).mean(axis=1).reshape(sample_size, sample_size)
# # #     en_err_sig = (monitor_dict['E0n'].Ierror[:, -last_t:]/pamp).mean(axis=1).reshape(sample_size, sample_size)
# # #
# # #     # plot input received at error neurons
# # #     fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
# # #     # Using matshow here just because it sets the ticks up nicely. imshow is faster.
# # #     input_plot = axes[0].matshow(en_input_sig, cmap='Reds')
# # #     fig2.colorbar(input_plot, ax=axes[0], shrink=0.6)
# # #     # for (i, j), z in np.ndenumerate(en_input_sig):
# # #     #     axes[0].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
# # #
# # #     # plot prediction received at error neurons
# # #     pred_plot = axes[1].matshow(en_pred_sig, cmap='Reds')
# # #     fig2.colorbar(pred_plot, ax=axes[1], shrink=0.6)
# # #     # for (i, j), z in np.ndenumerate(en_pred_sig):
# # #     #     axes[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
# # #
# # #     plt.show()
# # #
# # #     abs_err = np.abs(monitor_dict['E0n'].Ierror[:, -last_t:] / pamp).mean(axis=1)
# # #     if abs_err.mean() < dw_lim:
# # #         break
# # #
# # #     # postsynaptic variable :
# # #     hebb_post = np.einsum('ij,ik->ijk',
# # #                           w_ep.T,
# # #                           monitor_dict['P1'].x_trace[:, -last_t:])
# # #
# # #     for signs in ['p', 'n']:
# # #         # presynaptic variable :
# # #         hebb_pre = np.einsum('ij,ik->ijk',
# # #                              w_ep,
# # #                              monitor_dict['E0' + signs].x_trace[:, -last_t:])
# # #         hebb_res = np.tile(abs_err > dw_lim, nlayer_dict['P1']).reshape(nlayer_dict['P1'], nlayer_dict['E0' + signs]).T
# # #
# # #         hebb_val = lr/2 * hebb_post.mean(axis=2).T * hebb_pre.mean(axis=2) * hebb_res
# # #         if signs == 'n':
# # #             w_ep -= np.clip(hebb_val, -max_dw, max_dw)
# # #         elif signs == 'p':
# # #             w_ep += np.clip(hebb_val, -max_dw, max_dw)
# # #         w_ep = np.clip(w_ep, 0, wmax)
# # #
# # #     if irun < niter-1:
# # #         for dict_key, val in monitor_dict.items():
# # #             net.remove(val)
# # #             # net.remove(monitor_dict)
# # #         monitor_dict.clear()
