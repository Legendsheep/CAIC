import numpy as np
import random
import math


bias = np.zeros(1)
print(bias, bias[0])

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
