#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:03:08 2018

@author: anastasia
"""



import numpy as np
import h5py


import sys
sys.path.append('/Users/anastasia/Documents/Github/rand_tensor/')
from utils import *
from randtensor import *

import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from TVRegDiff import *

from sklearn.linear_model import LinearRegression

from pykalman import KalmanFilter


random.seed(1)

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






n1_neurons = list(group2['N1']['NeuronNames'][:])
for n in range(0, 109):
    myref = n1_neurons[n][0]
    object1 = f[myref]
    n1_neurons[n][0] = object1

n1_times = (group2['N1']['tv'][:])

n1_data = group2['N1']['processed']['deltaFOverF_bc_detr'][:]

n1_smoothed = group2['N1']['processed']['deltaFOverF_bc_detr_derivs'][:]

n1_PCA = list(group2['N1']['processed']['PCA']['PCs']) #length = 109

states = list(group2['N1']['States']) #['dt', 'fwd', 'nostate', 'prev1', 'rev2', 'revsus', 'slow', 'vt']

neuron_names = []

for i in range(len(group2['N1']['NeuronNames'])):

    name_i_h5_ref = group2['N1']['NeuronNames'][i][0]
    name_i_h5 = f[name_i_h5_ref]
    name = ''.join(chr(c) for c in name_i_h5[:])
    neuron_names.append(name)
print(neuron_names)




sizes = n1_data.shape
n1_mean = first_moment(n1_data)
n1_covs = second_moments(n1_data-n1_mean) #0 index is neurons, 1 index is time

n1_rand = randtensor(sizes)
n1_rand.fitMaxEntropy(n1_covs)
n1_surrogate = n1_rand.sampleTensors(1)




def PCA_3d(subject): #label axes?
    colors = ['orange', 'teal', 'white', 'red', 'pink', 'green', 'blue', 'yellow']
    n1_PCA = list(group2[subject]['processed']['PCA']['PCs'])


    PC1 = list(n1_PCA[0]) #3137
    PC2 = list(n1_PCA[1]) #3137
    PC3 = list(n1_PCA[2]) #3137
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for f in range(0,8):
        state = states[f]
        binary_state_list = list(group2[subject]['States'][state])
        index = np.ones(int(sum(binary_state_list)), dtype = int)
        counter = 0
        for x in range(0, len(binary_state_list)):
            if binary_state_list[x] > 0.0:
                index[counter] = x
                counter = counter + 1
        index = list(index)   
        PC1_graph = np.ones(int(sum(binary_state_list)))
        PC2_graph = np.ones(int(sum(binary_state_list)))
        PC3_graph = np.ones(int(sum(binary_state_list)))
    
        for i in range(0, len(index)):
            PC1_graph[i] = PC1[index[i]]
            PC2_graph[i] = PC2[index[i]]
            PC3_graph[i] = PC3[index[i]]
    
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.plot(PC1_graph, PC2_graph, PC3_graph, color = colors[f])
        
PCA_3d('N2')
        




#threshold_dict = {}



def AVA_phase(subject, neuron):
    phase_colors = ['blue', 'red', 'green', 'yellow'] 
    #low = blue, rise = red, high = green, fall = yellow
    neuron_index = neuron - 1 #index of AVA neuron
    x = group2[subject]['processed']['deltaFOverF_bc_detr'][neuron_index][:]
    #x = group2[subject]['processed']['deltaFOverF_bc_detr_derivs'][neuron_index][0:3000] #109x3137
    #x = TVRegDiff( x_raw, 300, 100)
    
     
    fps = group2[subject]['fps'][:]

    phase_array = group2[subject]['processed']['deltaFOverF_bc_detr_derivs'][neuron_index][:]
    
    rise_array = []
    fall_array = []
    
    
    range1 = max(phase_array) - min(phase_array)
         
    #thresholds = [range1/3-0.25, 2*range1/3-0.25]
    #thresholds = [-0.01, 0.036209657723480004]
    #if neuron in threshold_dict.keys():
    #    thresholds = threshold_dict[neuron]
    #else:
    thresholds = [-0.008, 0.015]
        
    index = 0    
    high = False
    step = 2
    
    length = len(phase_array)-1

    while index < length:
        if (phase_array[index] > thresholds[1]):
            plt.plot(range(index, index + step), x[index:index+step], color = 'red')
            high = True
            if (index > 0) & (phase_array[index-1] < thresholds[1]):
                rise_array.append(index)
            elif (index == 0):
                rise_array.append(index)
                
                
        elif (phase_array[index] < thresholds[0]):
            plt.plot(range(index, index + step), x[index:index+step], color = 'yellow')
            high = False
            if (index > 0) & (phase_array[index-1] > thresholds[0]):
                fall_array.append(index)
            elif (index == 0):
                fall_array.append(index)
            
            
        else:
            if high == True:
                plt.plot(range(index, index + step), x[index:index+step], color = 'green')
            #elif x[index] > 0.5:
            #    plt.plot(range(index, index + step), x[index:index+step], color = 'green')
            else:
                plt.plot(range(index, index + step), x[index:index+step], color = 'blue')
        
        index = index + 1

    
    plt.show()
    
    return rise_array, fall_array
    


AVA_phase('N1', 37)


#threshold_dict[neuron] = thresholds 
threshold_dict[41] = [-0.001, 0.003]



def phase_lags(array1, array2):
    phase_lag = []
    for a in array1:
        distance = 10000
        for b in array2:
            if abs(a-b) < abs(distance):
                distance = b-a
                
        phase_lag.append(distance)

    return phase_lag
        


def generate_hist(neuron):
    
    def percent(phase_lags):
        yes = 0
        for a in phase_lags:
            if (a > -20) and (a < 20):
                yes = yes + 1
            
        return round((yes/len(phase_lags)),3)*100
        
    
    n1_states = AVA_phase('N1', 36)
    n1_rise = n1_states[0]
    n1_fall = n1_states[1]
    
    
    n2_states = AVA_phase('N1', neuron)
    n2_rise = n2_states[0]
    n2_fall = n2_states[1]
    
    bins = np.linspace(-20,20,41)
    plt.close()
               
    n12rr = phase_lags(n1_rise, n2_rise)  
    


    plt.hist(n12rr, bins = bins)
    plt.title(str(neuron_names[neuron-1]) + ' Rise/Rise ' + str(percent(n12rr)))
    plt.savefig('/Users/anastasia/Documents/Github/rand_tensor/hist/' + str(neuron_names[neuron-1])+ 'rr.png')

    plt.close()
     
    n12ff = phase_lags(n1_fall, n2_fall)


    plt.hist(n12ff, bins = bins)
    plt.title(str(neuron_names[neuron-1]) + ' Fall/Fall ' + str(percent(n12ff)))
    plt.savefig('/Users/anastasia/Documents/Github/rand_tensor/hist/' + str(neuron_names[neuron-1])+ 'ff.png')

    plt.close()


    n12rf = phase_lags(n1_rise, n2_fall)

    plt.hist(n12rf, bins = bins)
    plt.title(str(neuron_names[neuron-1]) + ' Rise/Fall '+ str(percent(n12rf)))
    plt.savefig('/Users/anastasia/Documents/Github/rand_tensor/hist/' + str(neuron_names[neuron-1])+ 'rf.png')
    plt.close()


    n12fr = phase_lags(n1_fall, n2_rise)

    plt.hist(n12fr, bins = bins)
    plt.title(str(neuron_names[neuron-1]) + ' Fall/Rise '+ str(percent(n12fr)))
    plt.savefig('/Users/anastasia/Documents/Github/rand_tensor/hist/' + str(neuron_names[neuron-1])+ 'fr.png')

    plt.close()
    
    






def surrogate_lin_reg(subject, num_samples):
    
    #n1_data = group2[subject]['processed']['deltaFOverF_bc_detr'][:][:,0:200]
    n1_data = group2['N1']['processed']['deltaFOverF_bc_detr'][:][:,0:200]
    
    sizes = n1_data.shape
    n1_mean = first_moment(n1_data)
    n1_covs = second_moments(n1_data-n1_mean) #0 index is neurons, 1 index is time

    n1_rand = randtensor(sizes)
    n1_rand.fitMaxEntropy(n1_covs)
    n1_surrogate = n1_rand.sampleTensors(num_samples)
    
    
    r_squared= []

    
    for i in range(num_samples):
        
        sample = n1_surrogate[i]
        
        deriv = np.ones(n1_data.shape)
        for x in range(len(sample)):
            
            row_deriv = TVRegDiff(sample[x], 100,100)
            deriv[x] = row_deriv
        
        
        score = multi_lin_reg(deriv)
        
        r_squared.append(score)
        
        
    plt.hist(r_squared)    
        
    return r_squared
    

surrogate_lin_reg('N1', 10)





x_dot =  group2['N1']['processed']['deltaFOverF_bc_detr_derivs'][:]




def multi_lin_reg(x_dot):

    x_dot = x_dot[:][:,0:200]
    
    smooth_x = np.ones((x_dot.shape[0], x_dot.shape[1]))
    
    for row in range(len(x_dot)):
    
        smooth_row = np.cumsum(x_dot[row])
        smooth_row = smooth_row + x_dot[row][0]   #add x[0]
        smooth_x[row] = smooth_row
    
    
    scores = []
    
    for t in range(0, x_dot.shape[0]):
        
        model = LinearRegression()
        
        model.fit(smooth_x.T, x_dot[t,:])
        
        score = model.score(smooth_x.T, x_dot[t,:])
        
        scores.append(score)
        
    
    #plt.scatter(range(0,x_dot.shape[0]), scores)
    
    mean_score = sum(scores)/ len(scores)
    
    return mean_score
    
    #xhat = model.predict(x.T[0,:])

    #RSS = sum((x_dot-xhat)**2)
    #TSS = sum((x_dot-np.mean(x_dot))**2)
    #r_squared = 1- RSS/TSS
    #return r_squared
    
    
   
    


#offsets?
def kalman(smooth_x_dot):
    

    #smooth_x_dot = group2['N1']['processed']['deltaFOverF_bc_detr_derivs'][:][:,0:1000]
    smooth_x_dot = group2['N3']['processed']['deltaFOverF_bc_detr_derivs'][:]
    
    smooth_x = np.ones((smooth_x_dot.shape[0], smooth_x_dot.shape[1])) #observations corresponding to times [0...n_timesteps-1]
    
    for row in range(len(smooth_x_dot)):
        
        smooth_row = np.cumsum(smooth_x_dot[row][0:(smooth_x_dot.shape[1])])
        smooth_row = smooth_row + smooth_x_dot[row][0]   #add x[0]
        smooth_x[row] = smooth_row
        
    smooth_x = smooth_x.T
    
    kf = KalmanFilter(n_dim_state = 10, n_dim_obs = smooth_x_dot.shape[0]) 
    
    #loglikelihoods = [10, 50, 100, 500, 1000]
    #loglikelihoods = [5, 10, 15, 20]
    loglikelihoods = range(1, 51)
    estimates = np.ones(50)
    
    for i in range(len(loglikelihoods)):
        
        kf = kf.em(X=smooth_x, n_iter= 1, em_vars = 'all')
        estimates[i] = kf.loglikelihood(smooth_x)
     
    estimates3 = estimates
    
    plt.plot(estimates)
    plt.title('Log Likelihoods Worm 3')
    plt.xlabel('# of EM Steps')
               
    percent_difference = np.diff(estimates)
    
    for i in range(len(percent_difference)):
        percent_difference[i] = percent_difference[i]/estimates[i]
    
    #############   
    #blind state estimates
    n_dim_state = kf.transition_matrices.shape[0]
    n_timesteps = smooth_x.shape[0]
    blind_state_estimates = np.zeros((n_timesteps, n_dim_state))
    for t in range(n_timesteps-1):
        if t == 0:
            blind_state_estimates[t] = kf.initial_state_mean
        blind_state_estimates[t + 1] = (np.dot(kf.transition_matrices, blind_state_estimates[t]) + kf.transition_offsets[t])

    ########
    
    
    
    filtered_state_estimates = kf.filter(smooth_x)[0] #[n_timesteps, n_dim_obs]
    
    plt.plot(smooth_x[0])
    
    plt.plot(filtered_state_estimates[0])
    
    plt.plot(blind_state_estimates)
    
    print(kf.transition_matrices)
    
    print(kf.observation_matrices)
    
    #print(kf.transition_covariance)
    
    #print(kf.observation_covariance)
    
    #print(kf.transition_offsets)
    
    #print(kf.observation_offsets)
    
    #print(kf.initial_state_mean)
    
    #print(kf.initial_state_covariance)
    
    
    return kf.transition_matrices
    
    
    
    
    
    
def regression_fit():
    
    #training data
    train_x_dot = x_dot[:][:,0:1500]
    
    train_smooth_x = np.ones((train_x_dot.shape[0], train_x_dot.shape[1]))
    
    for row in range(len(train_x_dot)):
    
        smooth_row = np.cumsum(train_x_dot[row])
        smooth_row = smooth_row + train_x_dot[row][0]   #add x[0]
        train_smooth_x[row] = smooth_row
        
        
    #testing data
    
    test_x_dot = x_dot[:][:,1500:2800]
    
    test_smooth_x = np.ones((test_x_dot.shape[0], test_x_dot.shape[1]))
    
    for row in range(len(test_x_dot)):
    
        smooth_row = np.cumsum(test_x_dot[row])
        smooth_row = smooth_row + test_x_dot[row][0]   #add x[0]
        test_smooth_x[row] = smooth_row
    
    
    test_scores = []
    train_scores = []
    
    for t in range(0, train_x_dot.shape[0]):
        
        model = LinearRegression()
        
        model.fit(train_smooth_x.T, train_x_dot[t,:])
        
        train_score = model.score(train_smooth_x.T, train_x_dot[t,:])
        
        train_scores.append(train_score)
        
        test_score = model.score(test_smooth_x.T, test_x_dot[t,:])
        
        test_scores.append(test_score)
        
    
    train_r = sum(train_scores)/ len(train_scores)
    
    test_r = sum(test_scores)/ len(test_scores)
    
    print(train_r)
    
    print(test_r)

    
    
    
def kalman_fit(): 
    
    smooth_x_dot = group2['N1']['processed']['deltaFOverF_bc_detr_derivs'][:][:,0:1500]
    
    smooth_x = np.ones((smooth_x_dot.shape[0], smooth_x_dot.shape[1])) #observations corresponding to times [0...n_timesteps-1]
    

    
    for row in range(len(smooth_x_dot)):
        
        smooth_row = np.cumsum(smooth_x_dot[row][0:(smooth_x_dot.shape[1])])
        smooth_row = smooth_row + smooth_x_dot[row][0]   #add x[0]
        smooth_x[row] = smooth_row
        
        
    smooth_x = smooth_x.T
    
    kf = KalmanFilter(n_dim_state = 10, n_dim_obs = smooth_x_dot.shape[0]) 
    
    for i in range(30):
        
        kf = kf.em(X=smooth_x, n_iter= 1, em_vars = 'all')
        
    train_likelihood = kf.loglikelihood(smooth_x)
    
    
    #testing data
    
    test_smooth_x_dot = group2['N1']['processed']['deltaFOverF_bc_detr_derivs'][:][:,1500:3000]
        
    test_smooth_x = np.ones((smooth_x_dot.shape[0], smooth_x_dot.shape[1])) 
    
    for row in range(len(test_smooth_x_dot)):
        
        smooth_row = np.cumsum(test_smooth_x_dot[row][0:(test_smooth_x_dot.shape[1])])
        smooth_row = smooth_row + test_smooth_x_dot[row][0]   #add x[0]
        test_smooth_x[row] = smooth_row
        
    test_smooth_x = test_smooth_x.T    
    
    
    test_likelihood = kf.loglikelihood(test_smooth_x)
    
    
    
    
    
    



