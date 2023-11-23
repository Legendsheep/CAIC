import numpy as np
import random
import math
# from HDC_library import lookup_generate, encode_HDC_RFF, evaluate_F_of_x




# Generates random binary matrices of -1 and +1
# when mode == 0, all elements are generated with the same statistics (probability of having +1 always = 0.5)
# when mode == 1, all the probability of having +1 scales with an input "key" i.e., when the inputs to the HDC encoded are coded
# on e.g., 8-bit, we have 256 possible keys
def lookup_generate(dim, n_keys, mode = 1):
    arr = np.empty((n_keys,dim), int)
    if mode == 0:
        for i in range(n_keys):
            random_arr = [-1 + 2*(random.uniform(0, 1) <= .5) for x in range(dim)]
            arr[i] =  np.array([random_arr])
    else:
        B_in = math.log2(n_keys)
        for i in range(n_keys):
            random_arr = [-1 + 2*(random.uniform(0, 1) <= i/(2**(B_in) - 1)) for x in range(dim)]
            arr[i] =  np.array([random_arr])
        
    return arr.astype(np.int8)
    

# Performs "part" of the HDC encoding (only input encoding, position encoding and bundling), without the thresholding at the end.
# Returns H = bundle_along_features(P.L)
# img is the input feature vector to be encoded
# position_table is the random matrix of mode == 0
# grayscale_table is the input encoding LUT of mode == 1
# dim is the HDC dimensionality D
def encode_HDC_RFF(img, position_table, grayscale_table, dim,Ba):
    img_hv = np.zeros(dim, dtype=np.int16)
    container = np.zeros((len(position_table), dim))
    for pixel in range(len(position_table)):
        #Get the input-encoding and XOR-ing result:  
        encoded_input = grayscale_table[img[pixel]]
        hv = np.multiply(position_table[pixel], encoded_input)
        container[pixel, :] = hv*1
        
    img_hv = np.sum(container, axis = 0) % Ba #bundling without the cyclic step yet
    return img_hv





dim = 20000 # high dimension for higher averaging
n_keys = 16


# Test lookup_generate
 # lookup generate probability tester
weights = lookup_generate(dim,n_keys,0)
grayscale = lookup_generate(dim,n_keys,1)

p_weights = 1/2 + np.sum(weights)/(dim*n_keys)/2 # should be approx. equal to .5
assert(abs(p_weights - .5) < .005)

# the probability should increase with increasing index
p_gray = [0] * n_keys
for key in range(n_keys):
    p_gray[key] = 1/2 + np.sum(grayscale[key])/(dim)/2
copy_gray = p_gray.copy()
p_gray.sort()
assert(copy_gray == p_gray)


# Test encode_HDC_RFF
input_vec = np.arange(n_keys)
all_one = np.ones((n_keys,dim))

assert(np.equal(np.sum(grayscale, axis = 0) ,encode_HDC_RFF(input_vec,all_one,grayscale,dim)).all())
assert(np.equal(-1*np.sum(grayscale, axis = 0) ,encode_HDC_RFF(input_vec,-1*all_one,grayscale,dim)).all())



print("passed all tests. Ready for take-off")

