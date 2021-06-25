import numpy as np
import matplotlib.pyplot as plt

# base_dim = np.random.normal(2000.0, 800.0, (3, 3)) * 10 ** -12
base_dim = np.ones((3,3)) * 2000e-12

# shape : square
type_1 = np.copy(base_dim)
type_1[1,1] = 0

# shape : X
type_2 = np.copy(base_dim)
loc_list = [(0,1), (1,0), (1,2), (2,1)]
for loc in loc_list:
    type_2[loc] = 0

# shape : H
type_3 = np.copy(base_dim)
loc_list = [(0,1), (2,1)]
for loc in loc_list:
    type_3[loc] = 0

plt.subplot(131)
plt.imshow(type_1/1e-12, cmap='Reds', vmin=1000, vmax=3000)
plt.subplot(132)
plt.imshow(type_2/1e-12, cmap='Reds', vmin=1000, vmax=3000)
plt.subplot(133)
plt.imshow(type_3/1e-12, cmap='Reds', vmin=1000, vmax=3000)
plt.show()
