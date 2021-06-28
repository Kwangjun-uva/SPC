from mnist_data import createTestset, plot_testset
from rsa import rsa_analysis2
import matplotlib.pyplot as plt
import numpy as np

nSample = 10
nDigit = 10

# create a testset (mnist)
test_set_shuffled, digits, test_set_idx, label_set_shuffled, sample_set_idx = createTestset(nSample, nDigit)
# # create a testset (cifar)
# test_set_shuffled, label_set_shuffled, digits, test_set_idx, sample_set_idx = cifar10_grayscale_data.createTestset(nSample, nDigit)
plt.show()

# save data
np.save('figures/testset_data',
        test_set_shuffled)
np.savez('figures/testset_dict',
         label_set_shuffled=label_set_shuffled,
         digits=digits,
         test_set_idx=test_set_idx,
         sample_set_idx=sample_set_idx)

# plot images (mnist)
img_plot = plot_testset(test_set_shuffled, test_set_idx, nDigit, nSample)#, digits)
# # plot images (cifar)
# img_plot = cifar10_grayscale_data.plot_testset(test_set_shuffled, test_set_idx, digits)
plt.show()

# test the representational dissimilarity of testset
img_rdm_plot = rsa_analysis2([test_set_shuffled],
                             ['normalized images'],
                             test_set_idx, digits,
                             'figures/rdm_normImg',
                             nDigit, nSample)
plt.show()