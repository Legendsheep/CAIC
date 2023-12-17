from HDC_library import train_HDC_RFF, compute_accuracy, evaluate_F_of_x, lookup_generate
import numpy as np
import random

index_marker = 0

n_class = 2
N_train = 100
Y_train = np.array([random.uniform(0, 1) <= .5 for x in range(N_train)])
Y_train = Y_train.astype(int)
Dim = 8
HDC_cont_train = lookup_generate(Dim,N_train,0) # random data
HDC_cont_train[:,index_marker] = (Y_train-.5)*2 # perfect correlation between marker index and labels
HDC_cont_train = HDC_cont_train.astype(int)
gamma = 10**(2) # choose good estimator
D_b = 5



centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train, gamma, D_b)

for i in range(Dim): # check corrolation catch
    if i != index_marker:
        assert(abs(centroids[1][i]) < .01) 
    else:
        assert(abs(centroids[1][i] - 1) < .01)
assert(np.equal(centroids[0],-centroids[1]).all())# check pos cancer is equal to minus neg cancer
assert(compute_accuracy(HDC_cont_train,Y_train,centroids,biases) == 1)

assert((np.abs(np.array(centroids)-np.array(centroids_q)) < 2**(-D_b)).all())


gamma = 10**(-16) # set weights to zero

centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train, gamma, D_b)

for i in range(Dim): # check corrolation catch
    assert(abs(centroids[1][i]) < .01) 
assert(np.equal(centroids[0],-centroids[1]).all())# check pos cancer is equal to minus neg cancer
assert((np.abs(np.array(centroids)-np.array(centroids_q)) < 2**(-D_b)).all())

index_markers = [0,1,2]

n_class = 3
N_train = 200
Y_train = np.array([random.choice([0,1,2]) for x in range(N_train)])
Y_train = Y_train.astype(int)
Dim = 8
HDC_cont_train = lookup_generate(Dim,N_train,0) # random data
HDC_cont_train[:,index_markers[0]] = ((Y_train == 0)-.5)*2 # perfect correlation between marker index and labels
HDC_cont_train[:,index_markers[1]] = ((Y_train == 1)-.5)*2 # perfect correlation between marker index and labels
HDC_cont_train[:,index_markers[2]] = ((Y_train == 2)-.5)*2 # perfect correlation between marker index and labels
HDC_cont_train = HDC_cont_train.astype(int)
gamma = 10**(2) # choose good estimator
D_b = 5

centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train, gamma, D_b)

for cla in range(n_class):
    for i in range(Dim): # check corrolation catch
        if i in index_markers:
            if i == index_markers[cla]:
                assert(abs(centroids[cla][i] - .666) < .01) 
            else:
                assert(abs(centroids[cla][i] + .333) < .01)
        else:
            assert(abs(centroids[cla][i]) < .01)
assert(compute_accuracy(HDC_cont_train,Y_train,centroids,biases) == 1)
assert((np.abs(np.array(centroids)-np.array(centroids_q)) < 2**(-D_b)).all())

print("passed all tests. Ready for take-off")
# est = (np.matmul(HDC_cont_train,centroids[1]) + biases[0])
# eps = 1 - np.multiply(Y_train , est)
# print(np.dot(centroids[1],centroids[1])/2 + gamma * np.sum(np.square(eps))) # should be 1/2