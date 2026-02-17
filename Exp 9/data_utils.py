# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:28:57 2025

@author: Wetzel
"""

import numpy as np

# =========================
# Data configuration
# =========================

dimX = 6   # 6 joint angles (6-DoF robot)
dimY = 3   # We output end-effector position (x, y, z)

low = -np.pi/2
high = np.pi/2

num_data_points = 1000
noise = 0.01

num_anchors = 600
K_MAX = 5


# =========================
# DH Transformation
# =========================

def dh_matrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])


# =========================
# Robot Arm Forward Model
# =========================

def forward_kinematics(thetas):
    """
    thetas: array of shape (6,)
    returns: end-effector position (x, y, z)
    """

    # Example fixed DH parameters (general non-spherical wrist case)
    d = [0.3, 0.0, 0.0, 0.4, 0.0, 0.1]
    a = [0.0, 0.5, 0.3, 0.0, 0.0, 0.0]
    alpha = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]

    T = np.eye(4)

    for i in range(6):
        T = T @ dh_matrix(thetas[i], d[i], a[i], alpha[i])

    position = T[:3, 3]
    return position


def ground_truth(X):
    """
    X: shape (N, 6)
    Returns: shape (N, 3)
    """
    Y = np.array([forward_kinematics(x) for x in X])
    return Y.reshape(-1, dimY)


# =========================
# Data Sampling
# =========================

def sample_data(n=1000):
    data = np.random.uniform(low, high, size=(n, dimX))
    return data.reshape(-1, dimX)


def enforce_boundaries(pts):
    pts = np.clip(pts, low, high)
    return pts.reshape(-1, dimX)


def perturb_data(pts, eps):
    noise = np.random.uniform(-eps, eps, size=pts.shape)
    pts_new = pts + noise
    pts_new = enforce_boundaries(pts_new)
    return pts_new.reshape(-1, dimX)


# =========================
# Example Run
# =========================

if __name__ == "__main__":
    X = sample_data(num_data_points)
    Y = ground_truth(X)
    print("Sample input shape:", X.shape)
    print("Sample output shape:", Y.shape)
