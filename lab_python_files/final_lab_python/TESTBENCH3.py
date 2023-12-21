import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

Show_graphs = False # turns on/off nelder mead visualisation

def evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt):
    return - (beta_-1)**2 - 0**2 - (alpha_sp-2)**2, 0, 0

B_cnt = 8
load_simplex = False

NM_iter = 350 #Maximum number of iterations
STD_EPS = 0.002 #Threshold for early-stopping on standard deviation of the Simplex
#Contraction, expansion,... coefficients:
alpha_simp = 1 #* 0.5
gamma_simp = 2 #* 0.6
rho_simp = 0.5
sigma_simp = 0.5

#Initialize the Simplex or either load a pre-defined one (we will use a pre-defined one in the lab for reproducibility)
if load_simplex == False:
    Simplex = []
    N_p = 6
    for ii in range(N_p):
        alpha_sp = np.random.uniform(-5, 5)
        beta_ = np.random.uniform(-5, 5)
        gamma = np.random.uniform(-5, 5)
        simp_arr = np.array([gamma, alpha_sp, beta_])
        Simplex.append(simp_arr*1)  
        
    #np.savez("Simplex2.npz", data = Simplex)
            
else:
    print("Loading simplex")
    Simplex = np.load("lab_python_files/final_lab_python/Simplex2.npz", allow_pickle = True)['data']

a = np.arange(-5, 5,.0125)
b = np.arange(-5, 5,.0125)
X, Y = np.meshgrid(a, b)
h = (X-2)**2 + (Y-1)**2

# cs = plt.contour(a,b,h, levels=10, extend='both') #[.5,1, 2, 4, 8, 16, 32,64]

#Compute the cost F(x) associated to each point in the Initial Simplex
F_of_x = []
Accs = []
Sparsities = []
for init_simp in range(len(Simplex)):
    simp_arr = Simplex[init_simp] #Take Simplex from list
    gamma = simp_arr[0] #Regularization hyperparameter
    alpha_sp = simp_arr[1] #Threshold of accumulators
    beta_ = simp_arr[2] #incrementation step of accumulators
    ############### F(x) for Nelder-Mead with x = [gamma, alpha_sp, beta] ###################
    #The function "evaluate_F_of_x_2" performs:
    #    a) thresholding and encoding of bundled dataset into final HDC "ternary" vectors (-1, 0, +1)
    #    b) Training and testing the HDC system on "Nbr_of_trials" trials (with different random dataset splits)
    #    c) Returns lambda_1*Acc + lambda_2*Sparsity, Accuracy and Sparsity for each trials
    local_avg, local_avgre, local_sparse = evaluate_F_of_x(1, 0, 0, beta_, 0, gamma, alpha_sp, 0, 0, 0, 0, 0, B_cnt)
    F_of_x.append(1 - np.mean(local_avg)) #Append cost F(x)  
    Accs.append(np.mean(local_avgre))
    Sparsities.append(np.mean(local_sparse))
    ##################################   

#Transform lists to numpy array:
F_of_x = np.array(F_of_x) 
Accs = np.array(Accs)
Sparsities = np.array(Sparsities)
Simplex = np.array(Simplex)

objective_ = [] #Will contain the objective value F(x) for each simplex as the Nelder-Mead search goes on
STD_ = [] #Will contain the standard deviation of all F(x) as the Nelder-Mead search goes on

# For the details about the Nelder-Mead step, please refer to the course notes / reference, we are simply implementing that
for iter_ in range(NM_iter):
    plt.contour(a,b,h, levels=[.5,1, 2, 4, 8, 16, 32,64], extend='both') #[.5,1, 2, 4, 8, 16, 32,64]
    plt.scatter(Simplex[:,1],Simplex[:,2])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("s1")
    plt.ylabel("s2")
    if Show_graphs:
        plt.show()
    STD_.append(np.std(F_of_x))
    if np.std(F_of_x) < STD_EPS and 20 < iter_:
        break #Early-stopping criteria
    
    #1) sort Accs, Sparsities, F_of_x, Simplex, add best objective to array "objective_"
    index_sort = F_of_x.argsort()
    F_of_x = F_of_x[index_sort]
    Accs = Accs[index_sort]
    Sparsities = Sparsities[index_sort]
    Simplex = Simplex[index_sort]
    objective_.append(F_of_x[0])
    
    
    #2) average simplex x_0 
    
    x_0 = np.average(Simplex[:-1,:], axis= 0)
    
    #3) Reflexion x_r
    
    x_r = x_0 + alpha_simp * (x_0 - Simplex[-1,:])
    
    #Evaluate cost of reflected point x_r
    
    F_curr, acc_curr, sparse_curr = evaluate_F_of_x(1, 0, 0, x_r[2], 0, x_r[0], x_r[1], 0, 0, 0, 0, 0, B_cnt)
    F_curr, acc_curr, sparse_curr = 1 - np.mean(F_curr), np.mean(acc_curr), np.mean(sparse_curr)
    if F_curr < F_of_x[-2] and  F_of_x[0] <= F_curr:
        F_of_x[-1] = F_curr
        Simplex[-1,:] = x_r
        Accs[-1] = acc_curr
        Sparsities[-1] = sparse_curr
        rest = False
    else:
        rest = True
        
    if rest == True:
        #4) Expansion x_e
        if F_of_x[0] > F_curr:
            
            x_e = x_0 + gamma_simp * (x_r - x_0)
            
            #Evaluate cost of reflected point x_e
            
            F_exp, acc_exp, sparse_exp = evaluate_F_of_x(1, 0, 0, x_e[2], 0, x_e[0], x_e[1], 0, 0, 0, 0, 0, B_cnt)
            F_exp, acc_exp, sparse_exp = 1 - np.mean(F_exp), np.mean(acc_exp), np.mean(sparse_exp)
            if F_exp < F_curr:
                F_of_x[-1] = F_exp
                Simplex[-1,:] = x_e
                Accs[-1] = acc_exp
                Sparsities[-1] = sparse_exp
            else:
                F_of_x[-1] = F_curr
                Simplex[-1,:] = x_r
                Accs[-1] = acc_curr
                Sparsities[-1] = sparse_curr
    
        else: 
            #4) Contraction x_c
            if F_curr < F_of_x[-1]:
                x_c = x_0 + rho_simp * (x_r - x_0)
            else:
                x_c = x_0 + rho_simp * (Simplex[-1,:] - x_0)
                
            #Evaluate cost of contracted point x_c
            
            F_c, acc_c, sparse_c = evaluate_F_of_x(1, 0, 0, x_c[2], 0, x_c[0], x_c[1], 0, 0, 0, 0, 0, B_cnt)
            F_c, acc_c, sparse_c = 1 - np.mean(F_c), np.mean(acc_c), np.mean(sparse_c)
            if F_c < F_of_x[-1] and F_c < F_curr:
                F_of_x[-1] = F_c
                Simplex[-1,:] = x_c
                Accs[-1] = acc_c
                Sparsities[-1] = sparse_c
            else:
                #4) Shrinking
                for rep in range(1, Simplex.shape[0]):
                    x_rep = Simplex[0,:] + sigma_simp * (Simplex[rep,:] - Simplex[0,:])
                    F_of_x[rep], Accs[rep], Sparsities[rep] = evaluate_F_of_x(1, 0, 0, x_rep[2], 0, x_rep[0], x_rep[1], 0, 0, 0, 0, 0, B_cnt)
                    F_of_x[rep], Accs[rep], Sparsities[rep] = 1 - np.mean(F_of_x[rep]), np.mean(Accs[rep]), np.mean(Sparsities[rep])
                    Simplex[rep,:] = x_rep


################################## 
#At the end of the Nelder-Mead search and training, save Accuracy and Sparsity of the best cost F(x) into the ACCS and SPARSES arrays
idx = np.argsort(F_of_x)
F_of_x = F_of_x[idx]
Accs = Accs[idx]
Sparsities = Sparsities[idx]
Simplex = Simplex[idx, :]    
################################## 

# assert(abs(Simplex[0,0] - 0) < .05) # change 0 to gamma in F(x)
assert(abs(Simplex[0,1] - 2) < .05)
assert(abs(Simplex[0,2] - 1) < .05)

assert(abs(F_of_x[0]-1) < .05)

print("passed all tests. Ready for take-off")

