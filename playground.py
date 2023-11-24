import numpy as np
import random
import math
import matplotlib.pyplot as plt

Simplex = []
N_p = 11
for ii in range(N_p):
    alpha_sp = np.random.uniform(0, 1)
    gam_exp = np.random.uniform(-5, -1)
    beta_ = np.random.uniform(0, 2) 
    gamma = 10**gam_exp
    simp_arr = np.array([gamma, alpha_sp, beta_])
    Simplex.append(simp_arr*1) 

Simplex = np.array(Simplex)
cut = Simplex[:-1]
last = Simplex[-1]
x_0_simp = np.average(Simplex[:-1], axis= 0)
print(x_0_simp)
print(x_0_simp+last)


plt.figure(1)
plt.plot(np.arange(10), np.arange(10), '--')

plt.figure(2)
plt.plot(np.arange(10), np.arange(10), '--')
plt.show()
# bias = np.zeros(1)
# print(bias, bias[0])

# n_keys = 256
# dim = 8
# B_in = math.log2(n_keys)
# arr = np.empty((n_keys,dim), int)
# for i in range(n_keys):
#     random_arr = [-1 + 2*(random.uniform(0, 1) <= i/(2**(B_in) - 1)) for x in range(dim)]
#     arr[i] =  np.array([random_arr])

# arr2 = np.empty((n_keys,dim), int)
# for i in range(n_keys):
#     random_arr = [-1 + 2*(random.uniform(0, 1) <= .5) for x in range(dim)]
#     arr2[i] =  np.array([random_arr])

# for i in range(n_keys):
#     # print(arr[i])
#     # print(arr2[i])
#     print(np.multiply(arr[i],arr2[i]))
