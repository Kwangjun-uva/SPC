import pickle5 as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def os_plot(savedir):

    with open(savedir + '/weight_dict.pickle', 'rb') as w_file:
        weight_dict = pickle.load(w_file)

    for i, (w_name, weights) in enumerate(weight_dict.items()):
        if 'pc' in w_name:
            # shape of the weight matrix : nP(l-1), nP(l)
            w_shape_lm1, w_shape_l = weights.shape
            # x : square shape for represenation, y : square shape for plotting
            if w_name == 'pc1':
                x1, x2, y1, y2 = np.repeat(np.sqrt(weights.shape).astype(int),2)
            elif w_name == 'pc2':
                x1, x2, y1, y2 = [int(np.sqrt(w_shape_lm1)), int(np.sqrt(w_shape_lm1)), 32, 16]
            elif w_name == 'pc3':
                x1, x2, y1, y2 = [32, 16, int(np.sqrt(w_shape_l)), int(np.sqrt(w_shape_l))]

            # print (w_name, [x1, x2, y1, y2])
            # reshape current weights for square rep. over all samples
            curr_weights = weights.reshape(x1, x2, w_shape_l)

            # create an empty plot space on which all weight rep. are plotted
            plt_space = np.zeros(((x1 + 2) * y1, (x2 + 2) * y2))

            # plot by row x column
            for j in range(y1):
                # define the starting row position for the weights of current sample
                ix = (x1 + 2) * j + 1
                for k in range(y2):
                    # define the starting column position for the weights of current sample
                    iy = (x2 + 2) * k + 1
                    min_w = curr_weights[:, :, k + j * y2].min()
                    max_w = curr_weights[:, :, k + j * y2].max()
                    plt_space[ix:ix + x1, iy:iy + x2] = (curr_weights[:, :, k + j * y2] - min_w) / (max_w - min_w)

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            pp = ax.imshow(plt_space, cmap='Reds')
            # ax.set_aspect('auto')
            ax.set_yticks(np.arange(0, (x1 + 2) * (y1+1), (x1 + 2)))
            ax.set_yticklabels([])
            # ax.set_yticklabels(np.arange(0, y1 + 1, 1))
            ax.set_xticks(np.arange(0, (x2 + 2) * (y2+1), (x2 + 2)))
            ax.set_xticklabels([])
            # ax.set_xticklabels(np.arange(0, y2 + 1, 1))
            ax.grid(which='both', alpha=0.5)
            fig.colorbar(pp, cmap='Reds')
            fig.suptitle(w_name + 'weights')
            fig.tight_layout()
            plt.savefig('nD10nS1024nEP100/os_' + w_name)
            plt.close('all')


def k_sparseness(fs_np, img_id):

    p1_start_idx = 784 * 3
    p1_endidx    = 784 * 3 + 1024
    p2_start_idx = 784 * 3 + 1024 * 3
    p2_endidx    = 784 * 3 + 1024 * 3 + 512
    p3_start_idx = 784 * 3 + 1024 * 3 + 512 * 3
    p3_endidx    = 784 * 3 + 1024 * 3 + 512 * 3 + 256

    fs_img1 = fs_np[:, img_id]
    fs_img1_p1 = fs_img1[p1_start_idx:p1_endidx]
    fs_img1_p2 = fs_img1[p2_start_idx:p2_endidx]
    fs_img1_p3 = fs_img1[p3_start_idx:p3_endidx]

    fr_img1_p1 = fs_img1_p1[np.nonzero(fs_img1_p1)] * 2
    fr_img1_p2 = fs_img1_p2[np.nonzero(fs_img1_p2)] * 2
    fr_img1_p3 = fs_img1_p3[np.nonzero(fs_img1_p3)] * 2

    k_sparse_p1 = np.sum((fr_img1_p1 - fr_img1_p1.mean()) ** 4) / (len(fr_img1_p1) * fr_img1_p1.std()) - 3
    k_sparse_p2 = np.sum((fr_img1_p2 - fr_img1_p2.mean()) ** 4) / (len(fr_img1_p2) * fr_img1_p2.std()) - 3
    k_sparse_p3 = np.sum((fr_img1_p3 - fr_img1_p3.mean()) ** 4) / (len(fr_img1_p3) * fr_img1_p3.std()) - 3


    fig1, axs1 = plt.subplots(nrows=1, ncols=3)
    axs1[0].hist(fr_img1_p1, bins='auto', density=True, label='mean= {0:.2f}'.format(fr_img1_p1.mean()))
    axs1[0].set_title('pc1')
    axs1[1].hist(fr_img1_p2, bins='auto', density=True, label='mean= {0:.2f}'.format(fr_img1_p2.mean()))
    axs1[1].set_title('pc2')
    axs1[2].hist(fr_img1_p3, bins='auto', density=True, label='mean= {0:.2f}'.format(fr_img1_p3.mean()))
    axs1[2].set_title('pc3')
    fig1.suptitle('non-zero firing rate')
    for i in range(3):
        axs1[i].set_aspect('auto')
        axs1[i].legend(loc='upper right')
    plt.tight_layout()
    # plt.close(fig1)


    fig2, axs2 = plt.subplots(nrows=1, ncols=3, sharey=True)
    axs2[0].hist(fr_img1_p1, bins='auto', density=True, label='mean= {0:.2f}'.format(k_sparse_p1.mean()))
    axs2[0].set_title('pc1')
    axs2[1].hist(fr_img1_p2, bins='auto', density=True, label='mean= {0:.2f}'.format(k_sparse_p2.mean()))
    axs2[1].set_title('pc2')
    axs2[2].hist(fr_img1_p3, bins='auto', density=True, label='mean= {0:.2f}'.format(k_sparse_p3.mean()))
    axs2[2].set_title('pc3')
    fig2.suptitle('sparseness kurtosis')
    for i in range(3):
        axs2[i].set_aspect('auto')
        axs2[i].legend(loc='upper right')
    plt.tight_layout()
    # plt.close('all')

    return fig1, fig2

fs_np = np.load('sparseness/fs.npy')
# fr_plot, k_sparse_plot = k_sparseness(fs_np, img_id=1)
# plt.show()

def k_img_sel(fs_np):

    p1_start_idx = 784 * 3
    p1_endidx    = 784 * 3 + 1024
    p2_start_idx = 784 * 3 + 1024 * 3
    p2_endidx    = 784 * 3 + 1024 * 3 + 512
    p3_start_idx = 784 * 3 + 1024 * 3 + 512 * 3
    p3_endidx    = 784 * 3 + 1024 * 3 + 512 * 3 + 256

    fs_img1_p1 = fs_np[p1_start_idx:p1_endidx]
    fs_img1_p2 = fs_np[p2_start_idx:p2_endidx]
    fs_img1_p3 = fs_np[p3_start_idx:p3_endidx]

    fr_img1_p1 = fs_img1_p1[np.nonzero(fs_img1_p1)] * 2
    fr_img1_p2 = fs_img1_p2[np.nonzero(fs_img1_p2)] * 2
    fr_img1_p3 = fs_img1_p3[np.nonzero(fs_img1_p3)] * 2

    k_sparse_p1 = np.sum((fr_img1_p1 - fr_img1_p1.mean()) ** 4) / (len(fr_img1_p1) * fr_img1_p1.std()) - 3
    k_sparse_p2 = np.sum((fr_img1_p2 - fr_img1_p2.mean()) ** 4) / (len(fr_img1_p2) * fr_img1_p2.std()) - 3
    k_sparse_p3 = np.sum((fr_img1_p3 - fr_img1_p3.mean()) ** 4) / (len(fr_img1_p3) * fr_img1_p3.std()) - 3


    fig1, axs1 = plt.subplots(nrows=1, ncols=3)
    axs1[0].hist(fr_img1_p1, bins='auto', density=True, label='mean= {0:.2f}'.format(fr_img1_p1.mean()))
    axs1[0].set_title('pc1')
    axs1[1].hist(fr_img1_p2, bins='auto', density=True, label='mean= {0:.2f}'.format(fr_img1_p2.mean()))
    axs1[1].set_title('pc2')
    axs1[2].hist(fr_img1_p3, bins='auto', density=True, label='mean= {0:.2f}'.format(fr_img1_p3.mean()))
    axs1[2].set_title('pc3')
    fig1.suptitle('non-zero firing rate')
    for i in range(3):
        axs1[i].set_aspect('auto')
        axs1[i].legend(loc='upper right')
    plt.tight_layout()
    # plt.close(fig1)


    fig2, axs2 = plt.subplots(nrows=1, ncols=3, sharey=True)
    axs2[0].hist(fr_img1_p1, bins='auto', density=True, label='mean= {0:.2f}'.format(k_sparse_p1.mean()))
    axs2[0].set_title('pc1')
    axs2[1].hist(fr_img1_p2, bins='auto', density=True, label='mean= {0:.2f}'.format(k_sparse_p2.mean()))
    axs2[1].set_title('pc2')
    axs2[2].hist(fr_img1_p3, bins='auto', density=True, label='mean= {0:.2f}'.format(k_sparse_p3.mean()))
    axs2[2].set_title('pc3')
    fig2.suptitle('sparseness kurtosis')
    for i in range(3):
        axs2[i].set_aspect('auto')
        axs2[i].legend(loc='upper right')
    plt.tight_layout()
    # plt.close('all')

    return fig1, fig2

# fs_np = np.load('sparseness_2021_08_06_09_39/fs.npy')
# fr_plot, k_sparse_plot = k_sparseness(fs_np, img_id=1)
# plt.show()
