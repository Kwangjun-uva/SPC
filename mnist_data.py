# from InitParams_pc2ffn import *
# from rsa import reordering, rsa_analysis2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def create_mnist_set(nSample, nDigit, test_digits=None, shuffle=False):

    if test_digits is not None:
        digits = test_digits
    else:
        # test_set contains n randomly selected examples for each of the n digits (0-9)
        digits = np.random.choice(range(0, 10), nDigit, replace=False)
        digits.sort()

    # load mnist dataset from tf.keras
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    test_set = np.zeros((nDigit * nSample, np.multiply(*x_train[0].shape)))
    label_set = []

    for i in range(nDigit):

        rand_ids = np.random.choice(5000, nSample)
        curr_digit = x_train[np.where(y_train == digits[i])][rand_ids].reshape(nSample, np.multiply(*x_train[0].shape))
        norm_digits = curr_digit / np.linalg.norm(curr_digit, axis=1).reshape(nSample, 1)
        test_set[i*nSample:(i+1)*nSample] = 12000 * norm_digits + 1000

        label_set.append(y_train[np.where(y_train == digits[i])][rand_ids])

    label_set = np.ravel(label_set)

    test_set_idx = np.arange(nDigit * nSample)

    if shuffle:
        # shuffle the order
        np.random.shuffle(test_set_idx)
        test_set_shuffled = test_set[test_set_idx, :]
        label_set_shuffled = [label_set[i] for i in test_set_idx]

        return test_set_shuffled, digits, test_set_idx, label_set_shuffled

    else:
        return test_set, digits, test_set_idx, label_set

def reordering(mat, idx):
    len_mat = np.arange(len(idx))
    reorder = np.ravel([np.where(idx == i)[0] for i in len_mat])
    new_mat = mat[reorder]

    return new_mat

def plot_mnist_set(testset, testset_idx, nDigit, nSample):

    # fig.tight_layout()
    X = reordering(testset, testset_idx)
    # Plot images of the digits
    fig = plt.figure()

    if nSample > 10:
        width = 10
    else:
        width = nSample

    img = np.zeros((30 * nDigit, 30 * width))
    for i in range(nDigit):
        ix = 30 * i + 1
        for j in range(width):#n_img_per_row):
            iy = 30 * j + 1
            img[ix:ix + 28, iy:iy + 28] = X[i * nSample + j].reshape((28, 28))

    plt.imshow(img, cmap=plt.cm.Reds, vmin=1000e-12, vmax=4000e-12)
    plt.xticks([])
    plt.yticks([])
    plt.title('MNIST images: nDigits={0}, nSample={1}'.format(nDigit, nSample))
    # plt.savefig('figures/normalized_digits')

    return fig

# # create a testset (mnist)
# test_set_shuffled, digits, test_set_idx, label_set_shuffled = create_mnist_set(nSample=10, nDigit=5)
# plot_mnist_set(testset=test_set_shuffled, testset_idx=test_set_idx, nDigit=5, nSample=10)
# plt.show()
# def plot_L1rep(mat, testsetidx, nDigit, nSample):
#
#     X = reordering(mat, testsetidx)
#     img_xy = int(np.sqrt(nI))
#
#     fig = plt.figure()
#     img = np.zeros((30 * nDigit, 30 * nSample))
#     for i in range(nDigit):
#         ix = 30 * i + 1
#         for j in range(nSample):
#             iy = 30 * j + 1
#             img[ix:ix + img_xy, iy:iy + img_xy] = X[i * nSample + j].reshape((img_xy, img_xy))
#
#     plt.imshow(img, cmap=plt.cm.Reds)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title('L1 representations of MNIST images')
#     plt.savefig('figures/L1rep')
#     plt.show()
#
#     return fig