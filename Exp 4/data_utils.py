# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:28:57 2025

@author: Wetzel
"""

import numpy as np

# data ranges
dimX = 2
dimY = 2

low = -3 # if the data ranges are in different orders of magnitudes, normalize to ensure proper nearest neighbor finding
high = 3

num_data_points = 1000
noise = 0.01

num_anchors = 600 # max 0.6 * num_data_points, to leave room for val and test sets

K_MAX = 5      # Upper limit for k


# ground truth

def ground_truth(X):
    Y = np.array( (X[:,0]**3 - 2*X[:,0]*X[:,1]**2 + 5 * X[:,0] + 5* X[:,1] , X[:,1]**2 -2*X[:,0]*X[:,1]+3*X[:,0]-2*X[:,1])  ).T
    return Y.reshape(-1,dimY)

def sample_data(n=1000):
    data = np.random.uniform(low, high, size=(n,dimX))
    return data.reshape(-1,dimX) 

def enforce_boundaries(pts):
    return pts.reshape(-1,dimX) 

def perturb_data(pts, eps):
    # pts: shape (N, 2)
    
    # random perturbation
    noise = np.random.uniform(-eps, eps, size=pts.shape)
    pts_new = pts + noise

    # enforce boundaries
    pts_new = enforce_boundaries(pts_new)
    return pts_new.reshape(-1,dimX) 


