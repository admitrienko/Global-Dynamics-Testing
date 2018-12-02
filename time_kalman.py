#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:35:51 2018
@author: anastasia
"""

import numpy as np
import h5py
import sys
import time

import matplotlib.pyplot as plt
from pykalman import KalmanFilter

     
    
def time_kalman(x, n_dim_state=6, n_iters=10):
    #parameter: time series 
    kf = KalmanFilter(n_dim_state = n_dim_state, n_dim_obs = x.shape[0]) 

    x = x.T

    loglikelihoods = np.zeros(n_iters)

    em_times = np.zeros(n_iters)
    ll_times = np.zeros(n_iters)
    
    for i in range(n_iters):
        em_start_time = time.time()
        kf = kf.em(X=x, n_iter= 1, em_vars = ['transition_covariance', 'observation_covariance', 'initial_state_mean', 'initial_state_covariance'])
        em_end_time = time.time()

        em_times[i] = em_end_time - em_start_time

        ll_start_time = time.time()
        loglikelihoods[i] = kf.loglikelihood(x)
        ll_end_time = time.time()

        ll_times[i] = ll_end_time - ll_start_time
    
    return em_times, ll_times
    
    
if __name__== "__main__":
    
    filename = '/Users/sbittner/Documents/wormies/data/WT_data.mat'
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

    #x = group2['N1']['processed']['deltaFOverF_bc_detr'][:]
    x = np.array(group2['N1']['processed']['deltaFOverF_bc_detr'][:]);

    n_iters = 10

    n_time_pts_list = [500, 1000, 2000, 3000];
    n_dim_state_list = [2, 4, 6, 8, 10];
    #n_time_pts_list = [10, 50];
    #n_dim_state_list = [2, 4];

    n_time_pts_len = len(n_time_pts_list)
    n_dim_state_len = len(n_dim_state_list)

    em_times = np.zeros((n_time_pts_len, n_dim_state_len))
    ll_times = np.zeros((n_time_pts_len, n_dim_state_len))

    for i in range(n_time_pts_len):
        n_time_pts = n_time_pts_list[i]
        for j in range(n_dim_state_len):
            n_dim_state = n_dim_state_list[j]
            print('%d time points, %d latent dimensions' % (n_time_pts, n_dim_state))
            em_times_ij, ll_times_ij = time_kalman(x[:,:n_time_pts], n_dim_state, n_iters)
            em_times[i,j] = np.mean(em_times_ij);
            ll_times[i,j] = np.mean(ll_times_ij);
            print('EM took %.2f seconds and log-like took %.2f seconds.\n' % (em_times[i,j], ll_times[i,j]))

    np.savez('kalman_times.npz', em_times=em_times, ll_times=ll_times, \
                                 n_time_pts_list=n_time_pts_list, \
                                 n_dim_state_list=n_dim_state_list);



