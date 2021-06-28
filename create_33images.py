import numpy as np
import matplotlib.pyplot as plt

base_mean = 2000e-12
noise_var = 200e-12

# shape : square
def create_shapes(base_mean, noise_var, n_imgs_per_shape):

    loc_list = [[(1, 1)], # square
                [(0, 1), (1, 0), (1, 2), (2, 1)], # X
                [(0, 1), (2, 1)], # H
                [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)] # diamond
                ]

    n_shapes = len(loc_list)
    n_imgs_total = n_shapes * n_imgs_per_shape

    imgs = np.ones((n_imgs_total, 3, 3)) * base_mean + np.random.normal(0, 1, (n_imgs_total, 3, 3)) * noise_var

    for i, loc in enumerate(loc_list):
        for loc_i in loc:
            imgs[(slice(i * n_imgs_per_shape, (i+1) * n_imgs_per_shape),) + loc_i] = 0

    return imgs

def plot_imgset(img_set, nsample, nshape, batch_size):

    sample_set = []
    [sample_set.append(img_set[batch_size * i:batch_size * i + nsample ]) for i in range(nshape)]
    sample_set = np.array(sample_set).reshape(nsample * nshape, *img_set.shape[1:])

    fig, axs = plt.subplots(ncols=nsample, nrows=nshape, figsize=(3*nsample, 3*nshape))
    axs = axs.flatten()
    ind = 0
    for ax_i, ax in enumerate(axs):
        ax.imshow(sample_set[ax_i]/1e-12, cmap='Reds', vmin=1000, vmax=3000)
    fig.suptitle('training samples')

    return fig