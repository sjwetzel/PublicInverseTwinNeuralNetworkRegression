# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:28:57 2025

@author: Wetzel
"""

import numpy as np

# data ranges
dimX = 2
dimY = 1

low = -1 # if the data ranges are in different orders of magnitudes, normalize to ensure proper nearest neighbor finding
high = 1

num_data_points = 500
noise = 0.02

num_anchors = 300 # max 0.6 * num_data_points, to leave room for val and test sets

K_MAX = 5      # Upper limit for k


# ground truth

def ground_truth(X):
    Y = np.array(np.sqrt(1-X[:,0]**2 - X[:,1]**2)).T
    return Y.reshape(-1,dimY)

def sample_data(n=1000):
    xs, ys = [], []
    while len(xs) < n:
        x = np.random.uniform(low,high)
        y = np.random.uniform(low,high)
        if x*x + y*y < 0.99:
            xs.append(x)
            ys.append(y)
    data = np.concatenate([[xs], [ys]],axis=0).T
    return data.reshape(-1,dimX) 

def enforce_boundaries(pts):
    # compute radius
    r = np.linalg.norm(pts, axis=1)

    # any point with r > 1 gets projected back to the circle
    mask = r > 0.999
    pts[mask] = pts[mask] / (r[mask, None]+0.000001)

    return pts.reshape(-1,dimX) 

def perturb_data(pts, eps):
    # pts: shape (N, 2)
    
    # random perturbation
    noise = np.random.uniform(-eps, eps, size=pts.shape)
    pts_new = pts + noise

    # enforce boundaries
    pts_new = enforce_boundaries(pts_new)
    return pts_new.reshape(-1,dimX) 


