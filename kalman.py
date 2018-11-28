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

     
    
def kalman(x):
    #parameter: time series 
    
    kf = KalmanFilter(n_dim_state = 10, n_dim_obs = x.shape[0]) 
    
    x = x.T

    loglikelihoods = range(1, 51)
    estimates = np.ones(50)
    
    for i in range(len(loglikelihoods)):
        
        kf = kf.em(X=x, n_iter= 1, em_vars = 'all')
        estimates[i] = kf.loglikelihood(x)
        print(i)
    
    plt.plot(estimates)
    plt.title('Log Likelihoods Worm 3')
    plt.xlabel('# of EM Steps')
               
    
    #filtered_state_estimates = kf.filter(smooth_x)[0] 
    
    #plt.plot(smooth_x[0])
    
    #plt.plot(filtered_state_estimates[0])
    
    
    return kf.transition_matrices
    
    
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

        x = group2['N1']['processed']['deltaFOverF_bc_detr'][:]

    
        kalman(x)
    
    
    
    
    