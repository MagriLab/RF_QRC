#!/usr/bin/env python
# coding: utf-8

import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from skopt.plots import plot_convergence
from QRC.crc import *
import time
import numpy as np
from functools import partial
#Hyperparameter Optimization using Grid Search plus Bayesian Optimization


def val_fun (RVC_Noise_upt,esn,tikh,N_fo,N_fw,N_in,N_val,N_washout,U,U_washout,U_tv,Y_tv,Win,W,k,tikh_opt):
    val_fun = partial(RVC_Noise_upt,esn,tikh,N_fo,N_fw,N_in,N_val,N_washout,U,U_washout,U_tv,Y_tv,Win,W,k,tikh_opt)

    return val_fun


def validate(val,kernell,search_space,n_in,n_tot,esn,tikh,N_fo,N_fw,N_in,N_val,N_washout,U,U_washout,U_tv,Y_tv,Win,W,k,tikh_opt):

    #Gaussian Process reconstruction
    # b_e = GPR(kernel = kernell,
    #         normalize_y = True, #if true mean assumed to be equal to the average of the obj function data, otherwise =0
    #         n_restarts_optimizer = 2,  #number of random starts to find the gaussian process hyperparameters
    #         noise = 1e-10, # only for numerical stability
    #         random_state = 10) # seed

    val_func =  val_fun(val,esn,tikh,N_fo,N_fw,N_in,N_val,N_washout,U,U_washout,U_tv,Y_tv,Win,W,k,tikh_opt)

    #Bayesian Optimization
    res = skopt.gp_minimize(val_func,                         # the function to minimize
                    search_space,                      # the bounds on each dimension of x
                    #base_estimator       = b_e,        # GP kernel
                    acq_func             = "EI",       # the acquisition function
                    n_calls              = n_tot,      # total number of evaluations of f
                    #x0                   = x1,        # Initial grid search points to be evaluated at
                    n_initial_points      = n_in,       # the number of initial random initialization points
                    n_restarts_optimizer = 3,          # number of tries for each acquisition
                    random_state         = 10,         # seed
                    )
    return res


def RVC_Noise_upt(esn,tikh,N_fo,N_fw,N_in,N_val,N_washout,U,U_washout,U_tv,Y_tv,Win,W,k,tikh_opt,x):

    print_flag  = True
    esn.rho     = round(10**x[0],2)
    esn.epsilon = x[1]
    esn.sigma_in = x[2]
    # ti       = time.time()
    lenn     = tikh.size
    Mean     = np.zeros(lenn)

    #Train using tv: training+val
    Wout = esn.train(U_washout, U_tv, Y_tv,Win,W)[1]
    #print(Wout.shape)
    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):

        #select washout and validation
        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p : N_washout + p + N_val].copy()
        U_wash = U[            p : N_washout + p        ].copy()

        #print('1')

        #washout before closed loop
        xf = esn.open_loop(U_wash, np.zeros(esn.N_units),Win,W)[-1]
        #print(xf.shape,Wout.shape)

        for j in range(lenn):
            #Validate
            Yh_val   = esn.closed_loop(N_val-1, xf, Wout[j],Win,W)[0]
            Mean[j] += np.log10(np.mean((Y_val-Yh_val)**2))

    if k==0:
        print('closed-loop time:', time.time() - t1)

    #select optimal tikh
    a           = np.argmin(Mean)
    tikh_opt[k] = tikh[a]
    k          +=1

    #print for every set of hyperparameters
    if print_flag:
        print(k, ': Spectral radius, Input Scaling, Tikhonov, Leak rate, MSE:',
              esn.rho, esn.sigma_in, tikh_opt[k-1], esn.epsilon,  Mean[a]/N_fo)

    return Mean[a]/N_fo
