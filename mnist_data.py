# from InitParams_pc2ffn import *
# from rsa import reordering, rsa_analysis2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def pre_process(data_set, label_set, nDigit, nSample, classes):

    training_set = np.zeros((nDigit * nSample, np.multiply(*data_set[0].shape)))
    training_labels = []

    for i in range(nDigit):
        digit = data_set[np.where(label_set == classes[i])]
        rand_ids = np.random.choice(len(digit), nSample)
        curr_digit = digit[rand_ids].reshape(nSample, np.multiply(*data_set[0].shape))
        norm_digits = curr_digit / np.linalg.norm(curr_digit, axis=1).reshape(nSample, 1)
        training_set[i * nSample:(i + 1) * nSample] = 12000 * norm_digits + 600

        training_labels.append(label_set[np.where(label_set == classes[i])][rand_ids])

    training_labels = np.ravel(training_labels)

    return training_set, training_labels

def create_mnist_set(data_type, nSample, nDigit, test_digits=None, shuffle=False):
    if test_digits is not None:
        digits = test_digits
    else:
        # test_set contains n randomly selected examples for each of the n digits (0-9)
        digits = np.random.choice(range(0, 10), nDigit, replace=False)
        digits.sort()

    # load mnist dataset from tf.keras
    # data_type = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = data_type.load_data()

    training_set, training_labels = pre_process(x_train, y_train, nDigit, nSample, digits)
    test_set, test_labels = pre_process(x_test, y_test, nDigit, nSample, digits)
    training_set_idx = np.arange(nDigit * nSample)

    if shuffle:
        # shuffle the order
        np.random.shuffle(training_set_idx)
        training_set = training_set[training_set_idx, :]
        # test_set = test_set[training_set_idx, :]
        training_labels = [training_labels[i] for i in training_set_idx]
        # test_labels = [test_labels[i] for i in training_set_idx]

    return training_set, training_labels, test_set, test_labels, digits, training_set_idx

def reordering(mat, idx):
    len_mat = np.arange(len(idx))
    reorder = np.ravel([np.where(idx == i)[0] for i in len_mat])
    new_mat = mat[reorder]

    return new_mat


def plot_mnist_set(testset, testset_idx, nDigit, nSample, savefolder):
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
        for j in range(width):  # n_img_per_row):
            iy = 30 * j + 1
            img[ix:ix + 28, iy:iy + 28] = X[i * nSample + j].reshape((28, 28))

    plt.imshow(img, cmap="Reds", vmin=600e-12, vmax=4000e-12)
    plt.xticks([])
    plt.yticks([])
    plt.title('MNIST images: nDigits={0}, nSample={1}'.format(nDigit, nSample))
    plt.savefig(savefolder + '/normalized_samples')

    return fig