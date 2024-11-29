import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from torch.distributions import MultivariateNormal
# import argparse

from utils import generate_phit, generate_expAt, generate_points_on_circle

class k_nn(torch.nn.Module):
    def __init__(self, x_dim, u_dim, hidden_dim):
        super(k_nn, self).__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        
        self.activation = nn.ELU()
        
        self.layer_input = nn.Linear(self.x_dim + 1, self.hidden_dim, bias=True)
        self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.layer_3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_4 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.layer_5 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_6 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.layer_out = nn.Linear(self.hidden_dim, u_dim, bias=True)
                
    def forward(self, x,t):
        z_in = torch.concat((x,t),dim=2)
        h = self.layer_input(z_in)
        h_temp = self.activation(self.layer_1(h)) 
        h_temp = self.activation(self.layer_2(h_temp)) 
        h = h_temp + h 

        h_temp = self.activation(self.layer_3(h)) 
        h_temp = self.activation(self.layer_4(h_temp)) 
        h = h_temp + h 

        h_temp = self.activation(self.layer_5(h)) 
        h_temp = self.activation(self.layer_6(h_temp)) 
        h = h_temp + h 

        z_out = self.layer_out(h) 
        return z_out

if __name__ == '__main__':
    

    ## Parameters
    exp_sys = 'Highdim' # experiment system BrownianBridge, NoisyOscillator, NyquistJohnson, Highdim
    init_dist = 'Gaussian' # initial distribution Gaussian, Circle
    target_dist = '4G' # 2G, 4G, Circle
    N = 2000 # number of samples
    T = 1000 # number of time steps
    tf = 1.0 # time horizon


    n_sigma = 1 # noise level
    n = 4 # state dimension
    m = int(n/2) # control dimension

    # Set System matrices
    print("Generating system matrices")
    if exp_sys == 'BrownianBridge':
        A = torch.tensor([[0.0, 1.0],[0.0, 0.0]])
        B = torch.tensor([[0.0],[1.0]])
    elif exp_sys == 'NoisyOscillator':
        w = 5 # frequency
        A = torch.tensor([[0.0, w],[-w, 0.0]])
        B = torch.tensor([[0.0],[1.0]])
    elif exp_sys == 'NyquistJohnson':
        A = torch.tensor([[0.0, 1.0],[-1.0, -1.0]])
        B = torch.tensor([[0.0],[1.0]])
    elif exp_sys == 'Highdim':
        T_mat = torch.zeros((m, m))
        torch.diagonal(T_mat).fill_(2)
        torch.diagonal(T_mat, offset=1).fill_(-1)
        torch.diagonal(T_mat, offset=-1).fill_(-1)
        A = torch.cat((torch.cat((torch.zeros((m, m)), torch.eye(m)), dim=1), torch.cat((-T_mat, -torch.eye(m)), dim=1)), dim=0)
        B = torch.cat((torch.zeros((m, m)), torch.eye(m)), dim=0)
    else:
        raise ValueError('Invalid system name')
    
    # Generate initial and target distributions
    print("Generating initial and target distributions")
    if init_dist == 'Gaussian':
        x_0 = MultivariateNormal(torch.zeros(n), torch.eye(n)).sample((N,))
    elif init_dist == 'Circle':
        radius = 1
        center = (0, 0)  # Center of the circle
        x_0 = torch.tensor(generate_points_on_circle(radius, N, center),dtype=torch.float32)
    else:
        raise ValueError('Invalid initial distribution name')
    
    y = torch.zeros((N,n))
    if target_dist == '2G':
        mu_target = torch.ones(n)*6.0
        y[:int(N/2)] = MultivariateNormal(mu_target, torch.eye(n)).sample((int(N/2),))
        y[int(N/2):] = MultivariateNormal(-mu_target, torch.eye(n)).sample((int(N/2),)) 
    elif target_dist == '4G':
        mu_target = torch.ones(n)*6.0
        rot_vec = torch.tensor([1, -1] * (n // 2)) *2.5
        y[:int(N/4)] = MultivariateNormal(mu_target, torch.eye(n)).sample((int(N/4),))
        y[int(N/4):int(N/2)] = MultivariateNormal(-mu_target, torch.eye(n)).sample((int(N/4),))
        y[int(N/2):int(3*N/4)] = MultivariateNormal(mu_target*rot_vec, torch.eye(n)).sample((int(N/4),))
        y[int(3*N/4):] = MultivariateNormal(-mu_target*rot_vec, torch.eye(n)).sample((int(N/4),))
    elif target_dist == 'Circle':
        radius = 1
        center = (0, 0)  # Center of the circle
        y = torch.tensor(generate_points_on_circle(radius, N, center),dtype=torch.float32)*2
    else:
        raise ValueError('Invalid target distribution name')
    
    # Generate time steps
    t_N = torch.linspace(0.0, tf, T).repeat(N,1).reshape(N,T,1).permute(1,0,2) # shape (T,N,1)
    dt = t_N[1,0,0] - t_N[0,0,0]

    # Generate forward trajectory
    print("Generating forward trajectory")
    expAt = generate_expAt(A, t_N)
    exp1tAtrans = generate_expAt(A.T, 1-t_N)
    expA = generate_expAt(A, torch.ones_like(t_N))
    exp1tA = generate_expAt(A, 1-t_N)
    phi_t = generate_phit(t_N, n, A, B, dt, exp_sys)
    phi_1 = phi_t[-1].repeat(T, 1, 1, 1)
    phi_1_t = generate_phit(1-t_N, n, A, B, dt, exp_sys)
    # phi_1_t = phi_1 - torch.einsum('tnij,tnjk->tnik', exp1tA, torch.einsum('tnij,tnjk->tnik', phi_t, exp1tAtrans))

    mat_for_x = expAt - torch.einsum('tnij,tnjk->tnik', torch.einsum('tnij,tnjk->tnik', torch.einsum('tnij,tnjk->tnik', phi_t, exp1tAtrans), torch.linalg.inv(phi_1)), expA)

    mat_for_y = torch.einsum('tnij,tnjk->tnik', torch.einsum('tnij,tnjk->tnik', phi_t, exp1tAtrans), torch.linalg.pinv(phi_1))

    phi_t_exp1tAtrans = torch.einsum('tnij,tnjk->tnik', phi_t, exp1tAtrans)
    Sigma_t = phi_t - torch.einsum('tnij,tnjk->tnik', torch.einsum('tnij,tnjk->tnik', phi_t_exp1tAtrans, torch.linalg.pinv(phi_1)), phi_t_exp1tAtrans.transpose(-1,-2))

    Z_f = torch.zeros((T, N, n))
    for i in range(T):
        try:
            Z_f[i,:,:] = MultivariateNormal(torch.zeros(n), Sigma_t[i,0,:,:]).sample((N,))
        except:
            print(i)

    X_f = torch.einsum('tnij,tnj->tni', mat_for_x, x_0.unsqueeze(0).repeat(T,1,1)) + torch.einsum('tnij,tnj->tni', mat_for_y, y.unsqueeze(0).repeat(T,1,1)) + n_sigma * Z_f
    u_f = torch.einsum('tnij,tnj->tni',torch.einsum('tnij,tnjk->tnik', torch.einsum('ij,tnjk->tnik', B.T, exp1tAtrans), torch.linalg.pinv(phi_1_t)), y[None,:,:] - torch.einsum('tnij, tnj->tni', exp1tA, X_f))

    # Training the neural network
    ITERS = 10000
    BATCH_SIZE = 64
    LR = 1e-2
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(device)

    model = k_nn(n,m,32).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler_lr =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    for i in range(ITERS):
        idx = torch.randperm(N)[:BATCH_SIZE]
        X_train = X_f[:,idx,:]
        t_train = t_N[:,idx,:]  
        u_train = u_f[:,idx,:]
        k_value = model.forward(X_train.to(device),t_train.to(device)) 
        loss = criterion(k_value,u_train.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler_lr.step()
        
        with torch.no_grad():
            k_value = model.forward(X_f.to(device),t_N.to(device))
            loss = criterion(k_value,u_f.to(device))
            if  (i+1)==ITERS or (i+1)%100==0:  
                print("Iteration: %d/%d, loss = %.12f" %(i+1,ITERS,loss.item()))


    model.to(device='cpu')

    # Generate prediction trajectory
    W_p = torch.zeros((T, N, m))
    for i in range(T):
        W_p[i,:,:] = MultivariateNormal(torch.zeros(m), dt*torch.eye(m)).sample((N,))

    X_p = torch.zeros(T,N,n)
    # X_p[0] = torch.randn(N,n)
    X_p[0] = x_0
    for i in range(0,T-1):
        u_pred = model.forward(X_p[i,None,:,:],t_N[i,None,:,:])[0]
        X_p[i+1] = X_p[i] + (A @ X_p[i].T).T * dt + (B @ u_pred.T).T * dt + n_sigma * (B @ W_p[i].T).T

    # save the prediction trajectory

    np.save(f'exp_result_data/X_p_{exp_sys}_{init_dist}_to_{target_dist}.npy',X_p.detach().numpy())
    np.save(f'exp_result_data/X_f_{exp_sys}_{init_dist}_to_{target_dist}.npy',X_f.detach().numpy())
    # np.save(f'exp_result_data/u_f_{exp_sys}_{init_dist}_to_{target_dist}.npy',u_f.detach().numpy())
    np.save(f'exp_result_data/x_0_{exp_sys}_{init_dist}_to_{target_dist}.npy',x_0.detach().numpy())
    np.save(f'exp_result_data/y_{exp_sys}_{init_dist}_to_{target_dist}.npy',y.detach().numpy())


    