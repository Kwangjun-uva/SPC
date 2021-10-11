import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle5 as pickle

from tf_local import AdEx_Layer

def new_init(self, sim_directory,
             # neuron_model_constants,
             num_pc_layers, num_pred_neurons, num_stim, gist_num,
             w_mat, w_mat_init):

        """
        :param neuron_model_constants: dict. contains parameters of AdEx neuron.
        :param num_pc_layers: int. number of pc_layers.
        :param num_pred_neurons: list of int. number of prediction layers
        :param num_stim: int. size of stimulus.
        :param gist_num: int. size of gist.
        :param gist_connp: list of float.
        :param gist_maxw: list of int.
        """

        self.model_dir = sim_directory

        # network architecture
        self.n_pc_layer = num_pc_layers
        self.n_pred = num_pred_neurons
        self.n_gist = gist_num
        self.n_stim = num_stim

        # self.n_groups = num_pc_layers * 3 + 1
        self.neurons_per_group = [self.n_stim] * 3 + np.repeat([self.n_pred[:-1]], 3).tolist() + [self.n_pred[-1]] + [
            self.n_gist]
        self.n_variable = sum(self.neurons_per_group)

        # initial weight preparation
        self.w = w_mat
        self.w_init = w_mat_init

        # constant weight
        # weight update time interval
        self.l_time = None


AdEx_Layer.__init__ = new_init

# specify the folder
save_folder = input()

# load sse from previous training
with open(save_folder + '/sse_dict.pickle', 'rb') as sse_handle:
    sse_original = pickle.load(sse_handle)
AdEx_Layer.sse = sse_original

# load training dictionary
training_dict = {}
for i,j in list(np.load(save_folder + '/training_dict.npz').items()):
    training_dict[i] = j
locals().update(training_dict)

# load simulation and network parameters
with open(save_folder + '/sim_params_dict.pickle', 'rb') as sim_pm:
    sim_params = pickle.load(sim_pm)
locals().update(sim_params)

# load learned weights from previous training
with open(save_folder + '/weight_dict.pickle', 'rb') as wdict:
    w_mat = pickle.load(wdict)
# convert them to tensors
for key, grp in w_mat.items():
    w_mat[key] = tf.convert_to_tensor(grp)

# load learned weights from previous training
with open(save_folder + '/weight_init_dict.pickle', 'rb') as wdict:
    w_mat_init = pickle.load(wdict)
# convert them to tensors
for key, grp in w_mat_init.items():
    w_mat_init[key] = tf.convert_to_tensor(grp)

# training_set, training_labels, test_set, test_labels, digits, training_set_idx
training_set = np.load(save_folder + '/training_data.npy')
test_set = np.load(save_folder + '/test_data.npy')

n_stim = training_set.shape[1]

# test inference on subset of test data
test_samples = 256
test_n_sample = 16
test_iter_idx = int(test_set.shape[0]/n_shape/test_samples)

testing_set = test_set[::test_iter_idx]
test_labels = test_set_labels[::test_iter_idx]

# build network
adex_01 = AdEx_Layer(sim_directory=save_folder,
                     num_pc_layers=n_pc_layers,
                     num_pred_neurons=n_pred_neurons, num_stim=n_stim, gist_num=n_gist,
                     w_mat=w_mat, w_mat_init=w_mat_init)

# test the model performance with the full test set
# plotting is only for n_samples per digit
sse = adex_01.test_inference(data_set=testing_set, ndigit=n_shape, nsample=test_n_sample,
                             simul_dur=sim_dur, sim_dt=dt, sim_lt=learning_window,
                             train_or_test='test')
plt.show()

########################################################################################################################
# classification
########################################################################################################################
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from math import isqrt
import pandas as pd

p1_pred = (adex_01.w['pc1'] @
           adex_01.xtr_record[sum(adex_01.neurons_per_group[:3]): sum(adex_01.neurons_per_group[:4])]).numpy().T
p1 = adex_01.xtr_record[sum(adex_01.neurons_per_group[:3]): sum(adex_01.neurons_per_group[:4])].numpy().T
p2 = adex_01.xtr_record[sum(adex_01.neurons_per_group[:6]): sum(adex_01.neurons_per_group[:7])].numpy().T
p3 = adex_01.xtr_record[sum(adex_01.neurons_per_group[:9]): sum(adex_01.neurons_per_group[:10])].numpy().T
gp = adex_01.xtr_record[-n_gist:].numpy().T
pp = [testing_set, p1_pred, p1, p2, p3, gp]

def calculate_metrics(estimator, labels):

    # Calculate and print metrics
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))

# test different numbers of clusters
# clusters = [10, 16, 36, 64, 144, 256]
def test_nc(clusters, dataset, labels):
    for n_clusters in clusters:
        estimator = MiniBatchKMeans(n_clusters=n_clusters)
        X = MinMaxScaler().fit_transform(dataset)
        estimator.fit(X)
        # print cluster metrics
        calculate_metrics(estimator, X, labels)
        # determine predicted labels
        cluster_labels = infer_cluster_labels(estimator, labels)
        predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)
        # calculate and print accuracy
        print('Accuracy: {}\n'.format(metrics.accuracy_score(labels, predicted_Y)))

def infer_cluster_labels(kmeans, actual_labels):

    inferred_labels = {}
    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        # print(labels)
        # print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):

    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels

def is_square(i: int) -> bool:
    return i == isqrt(i) ** 2

# scaled inertia
def kMeansRes(scaled_data, k, alpha_k=0.02):
    '''
    Parameters
    ----------
    scaled_data: matrix
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns
    -------
    scaled_inertia: float
        scaled inertia value for current k
    '''

    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia

def chooseBestKforKMeans(scaled_data, k_range):
    ans = []
    for k in k_range:
        scaled_inertia = kMeansRes(scaled_data, k)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results

def kmeans_class(data, true_label, nc, which='all'):

    x = MinMaxScaler().fit_transform(data)
    kmeans = MiniBatchKMeans(n_clusters=nc)
    kmeans.fit(x)

    cluster_labels = infer_cluster_labels(kmeans, true_label)
    test_clusters = kmeans.predict(x)
    predicted_labels = infer_data_labels(test_clusters, cluster_labels)
    class_acc = metrics.accuracy_score(true_label, predicted_labels)

    xx, xy = [int(np.sqrt(x.shape[1]))] * 2
    # record centroid values
    centroids = kmeans.cluster_centers_

    # reshape centroids into images
    images = centroids.reshape(nc, xx, xy)

    # create figure with subplots using matplotlib.pyplot
    if is_square(nc):
        pltx, plty = [int(np.sqrt(nc))] * 2
    else:
        pltx, plty = [nc, 1]

    fig, axs = plt.subplots(pltx, plty, figsize=(20, 20))
    # loop through subplots and add centroid images
    for i, ax in enumerate(axs.flat):
        # determine inferred label using cluster_labels dictionary
        for key, value in cluster_labels.items():
            if i in value:
                ax.set_title('Inferred Label: {}'.format(key))

        # add image to subplot
        ax.imshow(images[i], cmap='Reds', vmin=0, vmax=1)
        ax.axis('off')
    fig.suptitle('classification centroids : n_components={0}, acc={1:.2f}'.format(nc, class_acc))

    return class_acc, fig

k_range = [10, 128]
best_ks = []
for i in range(len(pp)):
    scaled_data = MinMaxScaler().fit_transform(pp[i])
    best_k, results = chooseBestKforKMeans(scaled_data, k_range)
    best_ks.append(best_k)

# table1 = pd.DataFrame(data={'data set':['input', 'p1_pred', 'p1', 'p2', 'p3', 'g'],
#                             'k10':np.zeros(len(pp)), 'k64':np.zeros(len(pp)), 'k7':np.zeros(len(pp))})
# accs = {'k10':np.zeros(len(pp)), 'k64':np.zeros(len(pp)), 'k7':np.zeros(len(pp))}
# for key, grp in accs.items():
#     nc = int(re.findall(r'\d+', bbb[0])[0])
#     for i in range(100):
#         grp += [kmeans_accuracy(ppi, test_labels, nc) for ppi in pp]
#     grp /= 100
#
# fig, ax = plt.subplots()
# # hide axes
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')
# ax.table(cellText=table1.round(2).values, colLabels=table1.columns, loc='center')
# fig.tight_layout()
# plt.show()
#
# for key, grp in table1.items():
#     if key != 'data set':
#         table[key] = accs[key]

# ncs = [10, 64, 100]
# for j in range(len(ncs)):
#     for i in range(len(pp)):
#         figi = kmeans_acc(pp[i], test_labels, ncs[j])
#         figi.savefig('/home/kwangjun/PycharmProjects/SPC/2021_09_27_17_37_nD3nS1024nEP100/kmeans_clustering/k' + str(ncs[j]) + table1['data set'][i])