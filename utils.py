import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal


def generate_phit(t, n, A, B, dt, sys):
    """
    t, A, B are tensors, A, B are the system matrices, sys is the system name
    t shape is (T, N, 1), A shape is (n,n) B shape is (n,m), output shape is (T, N, n, n). this function generates all phi_t from t=0 to t=tf 
    """
    if sys == 'BrownianBridge':
        T, N, _ = t.shape
        phi_t = torch.zeros((T, N, n, n))
        phi_t[:,:,0,0] = t[:,:,0]**3/3
        phi_t[:,:,0,1] = t[:,:,0]**2/2
        phi_t[:,:,1,0] = t[:,:,0]**2/2
        phi_t[:,:,1,1] = t[:,:,0]
    elif sys == 'NoisyOscillator':
        w = A[0,1]
        T, N, _ = t.shape
        phi_t = torch.zeros((T, N, n, n))
        phi_t[:,:,0,0] = t[:,:,0]/2 - torch.sin(2*w*t[:,:,0])/(4*w)
        phi_t[:,:,0,1] = torch.sin(w*t[:,:,0])**2/(2*w)
        phi_t[:,:,1,0] = torch.sin(w*t[:,:,0])**2/(2*w)
        phi_t[:,:,1,1] = t[:,:,0]/2 + torch.sin(2*w*t[:,:,0])/(4*w)
    elif sys == 'NyquistJohnson':
        T, N, _ = t.shape
        phi_t = torch.zeros((T, N, n, n))
        phi_t[:,:,0,0] = (torch.exp(-t[:,:,0])*(torch.cos(torch.pi/3 + 3**(1/2)*t[:,:,0]) - 2))/3 + 1/2
        phi_t[:,:,0,1] = (1-torch.cos(3**(1/2)*t[:,:,0])) * torch.exp(-t[:,:,0])/3
        phi_t[:,:,1,0] = (1-torch.cos(3**(1/2)*t[:,:,0])) * torch.exp(-t[:,:,0])/3
        phi_t[:,:,1,1] = (torch.exp(-t[:,:,0])*(torch.cos(torch.pi/3 - 3**(1/2)*t[:,:,0]) - 2))/3 + 1/2
    elif sys == 'Highdim':
        T, N, _ = t.shape
        phi_t = torch.zeros((T, N, n, n))
        for i in range(T):
            phi_t[i] = generate_onetime_phit(A, B, t[i,0,0], dt, n, N)
    else:
        raise ValueError('Invalid system name')
    
    return phi_t


def generate_expAt(A, t):
    """
    A shape is (n,n), t shape is (T,N,1), output shape is (T,N,n,n)
    """
    T, N, _ = t.shape
    A_rep = A.repeat(T,N,1,1)
    A_t = A_rep * t.unsqueeze(-1)
    A_t_exp = torch.matrix_exp(A_t)

    return A_t_exp

def generate_onetime_phit(A, B, t, dt, n, N):
    """
    t is the scalar time, dt is the time step, output shape is (N, n, n), this function is used to generate the phi_t at time t
    """
    # phi_t = torch.zeros((N, n, n))
    idx = (t // dt + 1).int()
    # print(idx)
    integral_recorder = torch.zeros((idx, n, n))
    for i in range(idx-1):
        integral_recorder[i] = torch.matrix_exp((t - i * dt) * A) @ B @ B.T @ torch.matrix_exp((t - i * dt) * A.T) * dt
    phi_one_time_t = torch.sum(integral_recorder, dim=0)
    phi_one_time_t = phi_one_time_t.repeat(N, 1, 1)
    return phi_one_time_t

def generate_points_on_circle(radius, n, center=(0, 0)):
    """
    Generate n points equally spaced on a circle.
    
    Parameters:
        radius (float): Radius of the circle.
        n (int): Number of points.
        center (tuple): (x, y) coordinates of the circle's center.
    
    Returns:
        np.ndarray: Array of shape (n, 2) containing (x, y) coordinates.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)  # Equally spaced angles
    x = center[0] + radius * np.cos(angles)  # X-coordinates
    y = center[1] + radius * np.sin(angles)  # Y-coordinates
    return np.column_stack((x, y))
