# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:28:57 2025

@author: Wetzel
"""

import numpy as np

# data ranges
dimX = 3
dimY = 3

low = -np.pi/2 # if the data ranges are in different orders of magnitudes, normalize to ensure proper nearest neighbor finding
high = np.pi/2

num_data_points = 1000
noise = 0.01

num_anchors = 600 # max 0.6 * num_data_points, to leave room for val and test sets

K_MAX = 5      # Upper limit for k


# ground truth
def spatial2(theta, phi1, phi2, L1=1.0, L2=1.0):
    x = np.cos(theta)*(L1*np.cos(phi1)+L2*np.cos(phi1+phi2))
    y = np.sin(theta)*(L1*np.cos(phi1)+L2*np.cos(phi1+phi2))
    z = L1*np.sin(phi1)+L2*np.sin(phi1+phi2)
    return x, y, z

def ground_truth(X):
    Y = np.array( spatial2(X[:,0],X[:,1],X[:,2])).T
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


