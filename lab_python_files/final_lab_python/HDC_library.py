"""
Design of a Hyperdimensional Computing Circuit for Bio-signal Classification via Nelder-Mead optimization
and LS-SVM Training.

*HDC library*

Computer-Aided IC Design (B-KUL-H05D7A)

ir. Ali Safa, ir. Sergio Massaioli, Prof. Georges Gielen (MICAS-IMEC-KU Leuven)

(Author: A. Safa)
"""

import numpy as np
import random
import math
from sklearn.utils import shuffle

# Receives the HDC encoded test set "HDC_cont_test" and test labels "Y_test"
# Computes test accuracy w.r.t. the HDC prototypes (centroids) and the biases found at training time
def compute_accuracy(HDC_cont_test, Y_test, centroids, biases):
    Acc = 0
    n_class = np.max(Y_test) + 1
    for i in range(Y_test.shape[0]):
        received_HDC_vector = HDC_cont_test[i]
        all_resp = np.zeros(n_class)
        for cl in range(n_class):
            final_HDC_centroid = centroids[cl]
             #compute LS-SVM response
            response = np.dot(received_HDC_vector,final_HDC_centroid) + biases[cl]
            all_resp[cl] = response
        
        class_idx = np.argmax(all_resp)
        if class_idx == Y_test[i]:
            Acc += 1
            
    return Acc/Y_test.shape[0]


# Generates random binary matrices of -1 and +1
# when mode == 0, all elements are generated with the same statistics (probability of having +1 always = 0.5)
# when mode == 1, all the probability of having +1 scales with an input "key" i.e., when the inputs to the HDC encoded are coded
# on e.g., 8-bit, we have 256 possible keys
def lookup_generate(dim, n_keys, mode = 1, m0chance = .5):
    arr = np.empty((n_keys,dim), int)
    if mode == 0:
        for i in range(n_keys):
            random_arr = [-1 + 2*(random.uniform(0, 1) <= m0chance) for x in range(dim)]
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
def encode_HDC_RFF(img, position_table, grayscale_table, dim):
    img_hv = np.zeros(dim, dtype=np.int16)
    container = np.zeros((len(position_table), dim))
    for pixel in range(len(position_table)):
        #Get the input-encoding and XOR-ing result:  
        encoded_input = grayscale_table[img[pixel]]
        hv = np.multiply(position_table[pixel], encoded_input)
        container[pixel, :] = hv*1
        
    img_hv = np.sum(container, axis = 0) #bundling without the cyclic step yet
    return img_hv


# Train the HDC circuit on the training set : (Y_train, HDC_cont_train)
# n_class: number of classes
# N_train: number of data points in training set
# gamma: LS-SVM regularization
# D_b: number of bit for HDC prototype quantization
def train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train, gamma, D_b):
    centroids = []
    centroids_q = []
    biases_q = []
    biases = []
    for cla in range(n_class):
        Y_train_cla = Y_train.copy()
        pos_det = Y_train_cla == cla
        neg_det = Y_train_cla != cla
        Y_train_cla =  np.array(pos_det) + np.array(neg_det)*-1
        Y_train_cla = Y_train_cla.astype(int)

        #The steps below implement the LS-SVM training, check out the course notes, we are just implementing that
        #Beta.alpha = L -> alpha (that we want) 
        Beta = np.zeros((N_train+1, N_train+1)) #LS-SVM regression matrix
        #Fill Beta:
        
        Beta[0,1:] = Y_train_cla
        Omega = np.multiply(np.outer(Y_train_cla,Y_train_cla), np.matmul(HDC_cont_train,np.transpose(HDC_cont_train))) + 1/gamma * np.identity(N_train)
        # for i in range(1,N_train+1):
        #     Beta[i,0] = Y_train_cla[i-1]
        #     Beta[i,1:] = Omega[i-1,:]
        Beta[1:,0] = Y_train_cla
        Beta[1:,1:] = Omega
        #Target vector L:
            
        L = np.ones((N_train+1,1))
        L[0] = 0
        
        #Solve the system of equations to get the vector alpha:
            
        alpha = np.linalg.solve(Beta,L)
        # print(np.matmul(Beta,alpha))
        
        # Get HDC prototype for class cla, still in floating point
        
        weighting = np.multiply(Y_train_cla,alpha[1:,0])
        final_HDC_centroid = np.dot(np.transpose(HDC_cont_train),weighting)
        

        # Quantize HDC prototype to D_b-bit
        biggest_val = np.max(np.abs(final_HDC_centroid))
        max_val = np.abs(biggest_val)
        if max_val == 0:
            bit_req = 0
        else:
            bit_req = math.ceil(math.log2(max_val))
        final_HDC_centroid_q = np.round(final_HDC_centroid/2**(-D_b + bit_req),0)*2**(-D_b + bit_req) #check if this is correct later
        #Amplification factor for the LS-SVM bias
        fact = 1 
        if np.max(np.abs(final_HDC_centroid)) == 0:
            print("Kernel matrix badly conditionned! Ignoring...")
            centroids_q.append(np.ones(final_HDC_centroid_q.shape)) #trying to manage badly conditioned matrices, do not touch
            biases_q.append(10000)
        else:
            centroids_q.append(final_HDC_centroid_q*1)
            biases_q.append(alpha[0,0]*fact)
            
        centroids.append(final_HDC_centroid*1)
        biases.append(alpha[0,0])

        if (n_class == 2):
            if np.max(np.abs(final_HDC_centroid)) == 0:
                print("Kernel matrix badly conditionned! Ignoring...")
                centroids_q.append(-1*np.ones(final_HDC_centroid_q.shape)) #trying to manage badly conditioned matrices, do not touch
                biases_q.append(-1*10000)
            else:
                centroids_q.append(final_HDC_centroid_q*-1)
                biases_q.append(alpha[0,0]*fact*-1)
            centroids.append(final_HDC_centroid*-1)
            biases.append(alpha[0,0]*-1)
            break
        
    return centroids, biases, centroids_q, biases_q


# Evaluate the Nelder-Mead cost F(x) over "Nbr_of_trials" trials
# (HDC_cont_all, LABELS) is the complete dataset with labels
# beta_ is the output accumulator increment of the HDC encoder
# bias_ are the random starting value of the output accumulators
# gamma is the LS-SVM regularization hyper-parameter
# alpha_sp is the encoding threshold
# n_class is the number of classes, N_train is the number training points, D_b the HDC prototype quantization bit width
# lambda_1, lambda_2 define the balance between Accuracy and Sparsity: it returns lambda_1*Acc + lambda_2*Sparsity
def evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt,dim_HDC):
    gamma = 10**gamma
    local_avg = np.zeros(Nbr_of_trials)
    local_avgre = np.zeros(Nbr_of_trials)
    local_sparse = np.zeros(Nbr_of_trials)
    # if dim_HDC > 1000:
    #     dim_HDC = 1000
    # elif dim_HDC < 10:
    #     dim_HDC = 10
    #Estimate F(x) over "Nbr_of_trials" trials
    for trial_ in range(Nbr_of_trials): 
        # HDC_cont_all, LABELS = shuffle(HDC_cont_all, LABELS) # Shuffle dataset for random train-test split
        HDC_cont_all_cpy = HDC_cont_all * 1

        # Take training set
        
        # Apply cyclic accumulation with biases and accumulation speed beta_

        HDC_cont_all_cpy = (beta_ * HDC_cont_all_cpy + bias_)% (2**(B_cnt+1)) #bundling cyclic 
        updown = np.array((HDC_cont_all_cpy//2**(B_cnt)))
        HDC_cont_all_cpy = (HDC_cont_all_cpy)% (2**(B_cnt)) #bundling cyclic 
        
        # Ternary thresholding with threshold alpha_sp:
            
        one_index = HDC_cont_all_cpy - (2**(B_cnt-1)) > alpha_sp
        mone_index = HDC_cont_all_cpy - (2**(B_cnt-1)) < -alpha_sp
        # HDC_cont_all_cpy = np.array(one_index) * 1 + np.array(mone_index) * -1
        HDC_cont_all_cpy = np.multiply(np.array(one_index) * 1 + np.array(mone_index) * -1,-1 + 2*updown)

        HDC_cont_train_cpy = HDC_cont_all_cpy[:N_train] 
        HDC_cont_test_cpy = HDC_cont_all_cpy[N_train:]

            

        Y_train = LABELS[:N_train] - 1
        Y_train = Y_train.astype(int)
        
        # Train the HDC system to find the prototype hypervectors, _q meqns quantized
        centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train_cpy, gamma, D_b)
        
        # Do the same encoding steps with the test set

        # HDC_cont_test_cpy = HDC_cont_test_ * 1
        
        # # Apply cyclic accumulation with biases and accumulation speed beta_

        # HDC_cont_test_cpy = (beta_ * HDC_cont_test_cpy + bias_)% (2**B_cnt) #bundling cyclic 
        
        
        # # Ternary thresholding with threshold alpha_sp:
            
        # one_index = HDC_cont_test_cpy - (2**(B_cnt-1)) > alpha_sp
        # mone_index = HDC_cont_test_cpy - (2**(B_cnt-1)) < -alpha_sp
        # HDC_cont_test_cpy = np.array(one_index) * 1 + np.array(mone_index) * -1
        
        Y_test = LABELS[N_train:] - 1
        Y_test = Y_test.astype(int)
        
        # Compute accuracy and sparsity of the test set w.r.t the HDC prototypes
        Acc = compute_accuracy(HDC_cont_test_cpy, Y_test, centroids_q, biases_q)
        sparsity_HDC_centroid = np.array(centroids_q).flatten() 
        nbr_zero = np.sum((sparsity_HDC_centroid == 0).astype(int))
        # SPH = nbr_zero/(sparsity_HDC_centroid.shape[0])
        SPH = (nbr_zero-n_class*dim_HDC)/(1000) #dont think this is correct? -n_class*dim_HDC
        local_avg[trial_] = lambda_1 * Acc + lambda_2 * SPH #Cost F(x) is defined as 1 - this quantity
        local_avgre[trial_] = Acc
        local_sparse[trial_] = SPH
        
    return local_avg, local_avgre, local_sparse #, centroids[0]

def genetic_weights(D_HDC,imgsize_vector, Np,  grayscale_table, mut_fact =0, prev_table = None, prev_centroid = None):
    reuse_index = round(Np*D_HDC)
    position_table = lookup_generate(D_HDC, imgsize_vector, mode = 0) # n_keys x dim
    grayscale_table_cp = grayscale_table *1
    if not (prev_table is None):
        abs_centr = np.abs(prev_centroid)
        index_sort = abs_centr.argsort()
        prev_table_cp = prev_table*1
        prev_table_cp = prev_table_cp[:,index_sort]
        grayscale_table_cp = grayscale_table_cp[:,index_sort]
        position_table[:,:reuse_index] = prev_table_cp[:,:reuse_index]
        # children = children_maker(prev_table_cp[:,:reuse_index],imgsize_vector,reuse_index,reuse_index)
        # position_table[:,reuse_index:3*reuse_index] = children
    if mut_fact != 0:
        random_mut = lookup_generate(D_HDC, imgsize_vector, 0, 1-mut_fact)
        position_table = np.multiply(position_table,random_mut)
    return position_table , grayscale_table_cp

def children_maker(parents, imgsize_vector,reuse_index,n_children):
    children = np.zeros(imgsize_vector)
    i=0
    while i < n_children:
        parent1 = random.randint(0,reuse_index-1)
        parent2 = random.randint(0,reuse_index-1)
        if parent1 == parent2:
            continue
        else:
            slice_idx = random.randint(0,imgsize_vector-1)
            parents[:slice_idx,parent1]
            parents[slice_idx:,parent2]
            child1 =   np.concatenate((parents[:slice_idx,parent1], parents[slice_idx:,parent2]), axis=0)
            child2 =   np.concatenate((parents[:slice_idx,parent2], parents[slice_idx:,parent1]), axis=0)
            # genes_sel = np.random.choice([0, 1], size=imgsize_vector)
            # child = np.dot(genes_sel,parents[parent1]) + np.dot(1 - genes_sel,parents[parent2])
        children = np.concatenate((children,child1))
        children = np.concatenate((children,child2))
        i +=1
    children = children[30:]
    return children.reshape((30,n_children*2))

def PCA(matrix, nb):
    std = matrix.std(axis = 0)
    normilized = (matrix - matrix.mean(axis = 0)) / matrix.std(axis = 0)
    for i in  range(matrix.shape[1]):
        if std[i] == 0:
            normilized[:,i] = np.zeros(matrix.shape[0])
    cov = np.cov(normilized, rowvar = False)
    eigval, eigvec = np.linalg.eig(cov)
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]
    pcad =  np.matmul(normilized, eigvec[:,:])
    pcad[:,nb:] = np.zeros((matrix.shape[0],matrix.shape[1]-nb))
    return pcad
