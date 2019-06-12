#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:35:51 2018

@author: anastasia
"""

import numpy as np
import h5py
import sys
sys.path.append('/Users/anastasia/Documents/Github/rand_tensor/')

import matplotlib.pyplot as plt
from pykalman import KalmanFilter

     
    
def kalman(x, n_dim_state):
    #parameter: time series, 
    
    kf = KalmanFilter(n_dim_state = n_dim_state, n_dim_obs = x.shape[0]) 
    
    x = x.T

    #loglikelihoods = range(1, 51)
    #estimates = np.ones(50)
    
    #for i in range(len(loglikelihoods)):
    #    
    #    kf = kf.em(X=x, n_iter= 1, em_vars = 'all')
    #    estimates[i] = kf.loglikelihood(x)
    #    print(i)
    

    converged = False
    
    tol = 1e-8
    
    LL = []
    
    i = 0
    while converged == False:
        
        kf = kf.em(X=x, n_iter= 1, em_vars = 'all')
        LL.append(kf.loglikelihood(x))
        
        LLold = LL[i]

        if i<=2:

            LLbase = LL[i];

        elif ((LL[i]-LLbase) < (1+tol)*(LLold-LLbase)):

            converged = True
       
        i = i + 1
        
    return i - 1

        
    
    #plt.plot(estimates)
    #plt.title('Log Likelihoods Worm 3')
    #plt.xlabel('# of EM Steps')
               
    
    #filtered_state_estimates = kf.filter(smooth_x)[0] 
    
    #plt.plot(smooth_x[0])
    
    #plt.plot(filtered_state_estimates[0])
    
    
    #return kf.transition_matrices





def kalman_fit(x): 
    
    
    train = x[:,0:1500]
    
    kf = KalmanFilter(n_dim_state = 5, n_dim_obs = train.shape[0]) 
    
    train = train.T
    
    for i in range(10): #increase this
        
        kf = kf.em(X=train, n_iter= 1, em_vars = 'all')
        print(i)
        
    
    
    #testing data
    
    test = x[:,1500:3000]
    
    test = test.T
    
    mean = np.matmul(kf.observation_matrices, kf.smooth(test)[0].T) 
    
    for i in range(len(mean)):
        
        mean[:,i] = mean[:,i] + kf.observation_offsets
    
    r_squared = np.sum(np.square(mean - test.T)) 
    
    return r_squared
    
    
if __name__== "__main__":
    
        filename = '/Users/anastasia/Desktop/zimmer/WT_data.mat'
        f = h5py.File(filename,'r') 
        key0 = list(f.keys())[0]
        key1 = list(f.keys())[1]
        key2 = list(f.keys())[2]

        group0 = f[key0] # key is #refs#, group with 1184 members
        group0keys = list(group0.keys()) #(0-9, A-Z, a-z) + (b-t)

        group1 = f[key1] # key is #subsystem#, group with 1 member

        group1keys = list(group1.keys()) #MCOS dataset

        ##############

        group2 = f[key2] # key is WT_data, group with 5 member
        group2keys = list(group2.keys()) 
        group2_variables = list(group2['N1'].keys()) #['NeuronNames', 'Opts', 'States', 'dataset', 'deltaFOverF', 'deltaFOverF_bc', 'fps', 'processed', 'tv']

        #######

        x = group2['N1']['processed']['deltaFOverF_bc_detr'][0:1500]

    
        kalman(x)
        #kalman_fit(x)
    
    

    
    
    
    
    