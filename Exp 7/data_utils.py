# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:28:57 2025

@author: Wetzel
"""

import numpy as np

# data ranges
dimX = 2
dimY = 2

low = -np.pi/2 # if the data ranges are in different orders of magnitudes, normalize to ensure proper nearest neighbor finding
high = np.pi/2

num_data_points = 1000
noise = 0.01

num_anchors = 600 # max 0.6 * num_data_points, to leave room for val and test sets

K_MAX = 5      # Upper limit for k


# ground truth
def forward_kinematics(theta1, theta2, L1=1.0, L2=1.0):
    """
    Forward kinematics of a 2D 2-link robot arm.

    Parameters
    ----------
    theta1, theta2 : float or np.ndarray
        Joint angles (radians)
    L1, L2 : float
        Link lengths

    Returns
    -------
    x, y : float or np.ndarray
        End-effector position
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y

def ground_truth(X):
    Y = np.array( forward_kinematics(X[:,0],X[:,1])).T
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


