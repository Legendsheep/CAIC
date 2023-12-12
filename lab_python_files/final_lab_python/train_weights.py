"""
*TRAIN weights*

ir. Ali Safa, ir. Sergio Massaioli, Prof. Georges Gielen (MICAS-IMEC-KU Leuven)

(Author: A. Safa)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from HDC_library import lookup_generate, encode_HDC_RFF, evaluate_F_of_x,genetic_weights,PCA
plt.close('all')

"""
1) HDC_RFF parameters: DO NOT TOUCH
"""
##################################   
#Replace the path "WISCONSIN/data.csv" with wathever path you have. Note, on Windows, you must put the "r" in r'C:etc..'
dataset_path = 'lab_python_files/final_lab_python/WISCONSIN/data.csv' 
##################################   
imgsize_vector = 30 #Each input vector has 30 features
n_class = 2
D_b = 4 #We target 4-bit HDC prototypes
B_cnt = 8
maxval = 256 #The input features will be mapped from 0 to 255 (8-bit)
D_HDC = 600 #HDC hypervector dimension
portion = 0.6 #We choose 60%-40% split between train and test sets
Nbr_of_trials = 1 #Test accuracy averaged over Nbr_of_trials runs K-fold cross validation
N_tradeof_points = 20 #Number of tradeoff points - use 100 
N_fine = int(N_tradeof_points*0.4) #Number of tradeoff points in the "fine-grain" region - use 30
#Initialize the sparsity-accuracy hyperparameter search
lambda_fine = np.linspace(-0.2, 0.2, N_tradeof_points-N_fine)
lambda_sp = np.concatenate((lambda_fine, np.linspace(-1, -0.2, N_fine//2), np.linspace(0.2, 1, N_fine//2)))
N_tradeof_points = lambda_sp.shape[0]
    
"""
2) Load dataset: if it fails, replace the path "WISCONSIN/data.csv" with wathever 
path you have. Note, on Windows, you must put the "r" in r'C:etc..'
"""
DATASET = np.loadtxt(dataset_path, dtype = object, delimiter = ',', skiprows = 1)
X = DATASET[:,2:].astype(float)
LABELS = DATASET[:,1]
LABELS[LABELS == 'M'] = 1
LABELS[LABELS == 'B'] = 2
LABELS = LABELS.astype(float)
# X = X.T / np.max(X, axis = 1)
# # X, LABELS = shuffle(X.T, LABELS)
# X = X.T
X = PCA(X,5)
X = X.T / np.max(np.abs(X), axis = 1)
X = X.T
imgsize_vector = X.shape[1]
N_train = int(X.shape[0]*portion)

"""
3) Generate HDC LUTs and bundle dataset
"""



# position_table = lookup_generate(D_HDC, imgsize_vector, mode = 0) #weight for XOR-ing
HDC_cont_all = np.zeros((X.shape[0], D_HDC)) #Will contain all "bundled" HDC vectors
bias_ = 0 # -> INSERT YOUR CODE #generate the random biases once np.zeros(Nbr_of_trials)
  #Input encoding LUT
prev_centroid = None
prev_weights = None
prev_fx = 0
max_iter = 40
global_fx = .966
while prev_fx < .972:
    grayscale_table_best = lookup_generate(D_HDC, maxval, mode = 1)
    WORK = np.zeros(max_iter)
    for weightiter in range(max_iter):  
        position_table, grayscale_table = genetic_weights(D_HDC,imgsize_vector, 0,  grayscale_table_best , 0, prev_weights , prev_centroid)
        for i in range(X.shape[0]):
            # if i%100 == 0:
                # print(str(i) + "/" + str(X.shape[0]))
            HDC_cont_all[i,:] = encode_HDC_RFF((np.round((maxval/2 - 1))* X[i,:]).astype(int), position_table, grayscale_table, D_HDC)
        
        print("HDC bundling finished...")

        """
        4) Nelder-Mead circuit optimization and HDC training
        """
        ################################## 
        #Nelder-Mead parameters
        NM_iter = 350 #Maximum number of iterations
        STD_EPS = 0.002 #Threshold for early-stopping on standard deviation of the Simplex
        #Contraction, expansion,... coefficients:
        alpha_simp = 1 
        gamma_simp = 2
        rho_simp = 0.5
        sigma_simp = 0.5
        ################################## 

        ACCS = np.zeros(max_iter)
        SPARSES = np.zeros(max_iter)
        load_simplex = False # Keep it to true in order to have somewhat predictive results


        #Initialize the Simplex or either load a pre-defined one (we will use a pre-defined one in the lab for reproducibility)
        if load_simplex == False:
            OrigSimplex = []
            N_p = 11
            for ii in range(N_p):
                alpha_sp = np.random.uniform(0, 1) * ((2**B_cnt) / 2)
                gam_exp = np.random.uniform(-5, 1)
                beta_ = np.random.uniform(0, 2) * (2**B_cnt-1)/imgsize_vector
                gamma = gam_exp
                bias_ = np.random.uniform(0, 2) * (2**B_cnt-1)/imgsize_vector
                D_b = 6
                # dim_HDC = np.random.uniform(10, D_HDC)
                simp_arr = np.array([gamma, alpha_sp, beta_, bias_,D_b])
                OrigSimplex.append(simp_arr*1)     
                #np.savez("Simplex2.npz", data = Simplex)            
        else:
            print("Loading simplex")
            OrigSimplex = np.load("lab_python_files/final_lab_python/Simplex2.npz", allow_pickle = True)['data']

        AccsOrig = []
        SparsitiesOrig = []
        Centroids = []
        lambda_1 = 0
        lambda_2 = 0
        for init_simp in range(len(OrigSimplex)):
            simp_arr = OrigSimplex[init_simp] #Take Simplex from list
            gamma = simp_arr[0] #Regularization hyperparameter
            alpha_sp = simp_arr[1] #Threshold of accumulators
            beta_ = simp_arr[2] #incrementation step of accumulators
            bias_ = simp_arr[3]
            D_b = simp_arr[4]
            # dim_HDC = simp_arr[4]
            ############### F(x) for Nelder-Mead with x = [gamma, alpha_sp, beta] ###################
            #The function "evaluate_F_of_x_2" performs:
            #    a) thresholding and encoding of bundled dataset into final HDC "ternary" vectors (-1, 0, +1)
            #    b) Training and testing the HDC system on "Nbr_of_trials" trials (with different random dataset splits)
            #    c) Returns lambda_1*Acc + lambda_2*Sparsity, Accuracy and Sparsity for each trials
            local_avg, local_avgre, local_sparse,centroid = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt,D_HDC)
            AccsOrig.append(np.mean(local_avgre))
            SparsitiesOrig.append(np.mean(local_sparse))
            Centroids.append(centroid)

        print("Progress: " + str(weightiter+1) + "/" + str(max_iter))
        # F(x) = 1 - (lambda_1 * Accuracy + lambda_2 * Sparsity) : TO BE MINIMIZED by Nelder-Mead
        lambda_1 = 1 # Weight of Accuracy contribution in F(x)
        lambda_2 = 0 # Weight of Sparsity contribution in F(x): varies!


        Simplex = OrigSimplex
        #Compute the cost F(x) associated to each point in the Initial Simplex
        Accs = np.array(AccsOrig)
        Sparsities = np.array(SparsitiesOrig)
        F_of_x = 1 + (-lambda_1)*Accs +  (-lambda_2) * Sparsities
        # for init_simp in range(len(Simplex)):
        #     simp_arr = Simplex[init_simp] #Take Simplex from list
        #     gamma = simp_arr[0] #Regularization hyperparameter
        #     alpha_sp = simp_arr[1] #Threshold of accumulators
        #     beta_ = simp_arr[2] #incrementation step of accumulators
        #     bias_ = simp_arr[3]
        #     dim_HDC = simp_arr[4]
        #     ############### F(x) for Nelder-Mead with x = [gamma, alpha_sp, beta] ###################
        #     #The function "evaluate_F_of_x_2" performs:
        #     #    a) thresholding and encoding of bundled dataset into final HDC "ternary" vectors (-1, 0, +1)
        #     #    b) Training and testing the HDC system on "Nbr_of_trials" trials (with different random dataset splits)
        #     #    c) Returns lambda_1*Acc + lambda_2*Sparsity, Accuracy and Sparsity for each trials
        #     local_avg, local_avgre, local_sparse = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt,dim_HDC)
        #     F_of_x.append(1 - np.mean(local_avg)) #Append cost F(x)  
        #     Accs.append(np.mean(local_avgre))
        #     Sparsities.append(np.mean(local_sparse))
            ##################################   
        
        #Transform lists to numpy array:
        F_of_x = np.array(F_of_x) 
        Accs = np.array(Accs)
        Sparsities = np.array(Sparsities)
        Simplex = np.array(Simplex)
        Centroids  =np.array(Centroids)
        
        objective_ = [] #Will contain the objective value F(x) for each simplex as the Nelder-Mead search goes on
        STD_ = [] #Will contain the standard deviation of all F(x) as the Nelder-Mead search goes on
        
        # For the details about the Nelder-Mead step, please refer to the course notes / reference, we are simply implementing that
        for iter_ in range(NM_iter):

            STD_.append(np.std(F_of_x))
            if np.std(F_of_x) < STD_EPS and 10 < iter_:
                break #Early-stopping criteria
            
            #1) sort Accs, Sparsities, F_of_x, Simplex, add best objective to array "objective_"
            index_sort = F_of_x.argsort()
            F_of_x = F_of_x[index_sort]
            Accs = Accs[index_sort]
            Sparsities = Sparsities[index_sort]
            Simplex = Simplex[index_sort]    
            Centroids = Centroids[index_sort,:]
            objective_.append(F_of_x[0])
            
            
            #2) average simplex x_0 
            
            x_0 = np.average(Simplex[:-1,:], axis= 0)
            
            #3) Reflexion x_r
            
            x_r = x_0 + alpha_simp * (x_0 - Simplex[-1,:])
            
            #Evaluate cost of reflected point x_r
            
            F_curr, acc_curr, sparse_curr,centroid_curr = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, x_r[2], x_r[3], x_r[0], x_r[1], n_class, N_train, x_r[4], lambda_1, lambda_2, B_cnt, D_HDC)
            F_curr, acc_curr, sparse_curr = 1 - np.mean(F_curr), np.mean(acc_curr), np.mean(sparse_curr)
            if F_curr < F_of_x[-2] and  F_of_x[0] <= F_curr:
                F_of_x[-1] = F_curr
                Simplex[-1,:] = x_r
                Accs[-1] = acc_curr
                Sparsities[-1] = sparse_curr
                Centroids[-1] = centroid_curr
                rest = False
            else:
                rest = True
                
            if rest == True:
                #4) Expansion x_e
                if F_of_x[0] > F_curr:
                    
                    x_e = x_0 + gamma_simp * (x_r - x_0)
                    
                    #Evaluate cost of reflected point x_e
                    
                    F_exp, acc_exp, sparse_exp,centroid_exp = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, x_e[2], x_e[3], x_e[0], x_e[1], n_class, N_train, x_e[4], lambda_1, lambda_2, B_cnt, D_HDC)
                    F_exp, acc_exp, sparse_exp = 1 - np.mean(F_exp), np.mean(acc_exp), np.mean(sparse_exp)
                    if F_exp < F_curr:
                        F_of_x[-1] = F_exp
                        Simplex[-1,:] = x_e
                        Accs[-1] = acc_exp
                        Sparsities[-1] = sparse_exp
                        Centroids[-1] = centroid_exp
                    else:
                        F_of_x[-1] = F_curr
                        Simplex[-1,:] = x_r
                        Accs[-1] = acc_curr
                        Sparsities[-1] = sparse_curr
                        Centroids[-1] = centroid_curr
        
                else: 
                    #4) Contraction x_c
                    if F_curr < F_of_x[-1]:
                        x_c = x_0 + rho_simp * (x_r - x_0)
                    else:
                        x_c = x_0 + rho_simp * (Simplex[-1,:] - x_0)
                    
                    #Evaluate cost of contracted point x_c
                    
                    F_c, acc_c, sparse_c,centroid_c = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, x_c[2], x_c[3], x_c[0], x_c[1], n_class, N_train, x_c[4], lambda_1, lambda_2, B_cnt, D_HDC)
                    F_c, acc_c, sparse_c = 1 - np.mean(F_c), np.mean(acc_c), np.mean(sparse_c)
                    if F_c < F_of_x[-1] and F_c < F_curr:
                        F_of_x[-1] = F_c
                        Simplex[-1,:] = x_c
                        Accs[-1] = acc_c
                        Sparsities[-1] = sparse_c
                        Centroids[-1] = centroid_c
                    else:
                        #4) Shrinking
                        for rep in range(1, Simplex.shape[0]):
                            x_rep = Simplex[0,:] + sigma_simp * (Simplex[rep,:] - Simplex[0,:])
                            F_of_x[rep], Accs[rep], Sparsities[rep],Centroids[rep] = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, x_rep[2], x_rep[3], x_rep[0], x_rep[1], n_class, N_train, x_rep[4], lambda_1, lambda_2, B_cnt, D_HDC)
                            F_of_x[rep], Accs[rep], Sparsities[rep] = 1 - np.mean(F_of_x[rep]), np.mean(Accs[rep]), np.mean(Sparsities[rep])
                            Simplex[rep,:] = x_rep

        ################################## 
        #At the end of the Nelder-Mead search and training, save Accuracy and Sparsity of the best cost F(x) into the ACCS and SPARSES arrays
        idx = np.argsort(F_of_x)
        F_of_x = F_of_x[idx]
        Accs = Accs[idx]
        Sparsities = Sparsities[idx]
        Simplex = Simplex[idx, :]    
        Centroids = Centroids[idx,:]
        WORK[weightiter] = Accs[0]
        print(Accs[0])
        SPARSES[weightiter] = Sparsities[0]
        ##################################        
        if Accs[0] > prev_fx:
            prev_centroid = Centroids[0]
            prev_weights = position_table
            prev_fx = Accs[0]
            grayscale_table_best = grayscale_table
        if prev_fx < .95:
            break
        
    if prev_fx > global_fx:
        import csv
        with open("weigth.csv","w",newline="") as file:
            writer = csv.writer(file)
            for k in range(imgsize_vector):
                writer.writerow(prev_weights[k])
        with open("enc.csv","w",newline="") as file:
            writer = csv.writer(file)
            for k in range(2**B_cnt):
                writer.writerow(grayscale_table_best[k])
        global_fx = prev_fx
"""
Plot results (DO NOT TOUCH CODE)
Your code above should return:
    SPARSES: array with sparsity of each chosen lambda_
    ACCS: array of accuracy of each chosen lambda_
    objective_: array of the evolution of the Nelder-Mead objective of the last lambda_ under test
    STD_: array of the standard deviation of the simplex of the last lambda_ under test
    
"""
#Plot tradeoff curve between Accuracy and Sparsity
# SPARSES_ = SPARSES[SPARSES > -100] 
# ACCS_ = ACCS[SPARSES > -100]
plt.figure(1)
plt.plot([x for x in range(max_iter)], WORK, 'x', markersize = 10)
plt.grid('on')
plt.xlabel("Iter")
plt.ylabel("Accuracy")

#Plot the evolution of the Nelder-Mead objective and the standard deviation of the simplex for the last run
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(objective_, '.-')   
plt.title("Objective")
plt.grid("on")
plt.subplot(2,1,2)
plt.plot(STD_, '.-') 
plt.title("Standard deviation") 
plt.grid("on")

# plt.figure(3)
# plt.plot(lambda_sp, ACCS)

# plt.figure(4)
# plt.plot(lambda_sp, SPARSES)

plt.show()


