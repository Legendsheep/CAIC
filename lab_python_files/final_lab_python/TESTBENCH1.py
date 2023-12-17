from HDC_library import lookup_generate, encode_HDC_RFF, evaluate_F_of_x
import numpy as np


dim = 200000 # high dimension for higher averaging
n_keys = 16


# Test lookup_generate
 # lookup generate probability tester
weights = lookup_generate(dim,n_keys,0)
grayscale = lookup_generate(dim,n_keys,1)

assert(weights.shape[0] == n_keys)
assert(weights.shape[1] == dim)
assert(grayscale.shape[0] == n_keys)
assert(grayscale.shape[1] == dim)
p_weights = 1/2 + np.sum(weights)/(dim*n_keys)/2 # should be approx. equal to .5
assert(abs(p_weights - .5) < .005)


# the probability should increase with increasing index
p_gray = [0] * n_keys
for key in range(n_keys):
    p_gray[key] = 1/2 + np.sum(grayscale[key])/(dim)/2
    assert(abs(p_gray[key] - key/(n_keys - 1)) < .005)


# Test encode_HDC_RFF
input_vec = np.arange(n_keys)
all_one = np.ones((n_keys,dim))

assert(np.equal(np.sum(grayscale, axis = 0) ,encode_HDC_RFF(input_vec,all_one,grayscale,dim)).all())
assert(np.equal(-1*np.sum(grayscale, axis = 0) ,encode_HDC_RFF(input_vec,-1*all_one,grayscale,dim)).all())

# encode_HDC_RFF(input_vec,all_one,grayscale,dim,4,4)

# dataset_path = 'WISCONSIN/data.csv' 
# DATASET = np.loadtxt(dataset_path, dtype = object, delimiter = ',', skiprows = 1)

# N_train = 88

# LABELS = DATASET[:,1]
# LABELS[LABELS == 'M'] = 1
# LABELS[LABELS == 'B'] = 2
# LABELS = LABELS.astype(float)

# Y_train = LABELS[:N_train] - 1

# a = np.multiply(np.outer(Y_train,Y_train),np.outer(Y_train,Y_train))

# print(a)
# print(np.ones((N_train+1,1)))


print("passed all tests. Ready for take-off")

