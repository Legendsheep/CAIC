from HDC_library import train_HDC_RFF, compute_accuracy, evaluate_F_of_x, lookup_generate
import numpy as np

n_class = 2
Y_train = np.array([-1,1,-1,1,1,-1,1,1,-1,1])
N_train = Y_train.size 
Dim = 8
HDC_cont_train = lookup_generate(Dim,N_train,1)
gamma = 10**(-2)
D_b = 5



train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train, gamma, D_b)