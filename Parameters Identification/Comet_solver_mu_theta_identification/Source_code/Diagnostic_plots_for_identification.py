##############################################################################################################
#
# - mu u_xx - mu u_yy + 10 cos(theta) u_x + 10 sin(theta) u_y = 10 exp(-100 |x - x_0|)   in \Omega = (0, 1)^2
# u(x,y) = 0                                                                             on \partial\Omega
#
###############################################################################################################



# %% Import relevant libraries

import nisaba as ns
import nisaba.experimental as nse
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json

import matplotlib.pyplot as plt
from matplotlib import cm



# %% Settings

domain_dim = 2
params_dim = 2

x_0 = np.array([0.5,0.5])

boundary  = lambda x: 0 * x[:,0]
forcing   = lambda x: 10 * np.exp(-100 * np.sqrt( np.power(x[:,0] - x_0[0], 2) + np.power(x[:,1] - x_0[1], 2) ))


mu    = 1.4444
theta = 0.66139



# %% Load model and solution to be reversed

# NN pretrained model

pretrained_model = ns.utils.import_ANN('current_model')

for layer in pretrained_model.layers:
        layer.trainable = False
        
# Minimum and maximum values of the numerical solutions of the training set
        
with open('current_minmax.json', 'r') as fp:
    minmax_bounds = json.load(fp)
    
min_train = minmax_bounds['min_train']
max_train = minmax_bounds['max_train']
    
# Solution for a given combination of parameters

path_to_tables = os.path.join(os.path.dirname(os.getcwd()), 'tables_mu_theta')

data_int = pd.read_csv (os.path.join(path_to_tables, 'tab_int.csv'), names = ('x','y','mu','theta','u'))
data_bc  = pd.read_csv (os.path.join(path_to_tables, 'tab_bc.csv' ), names = ('x','y','mu','theta','u'))


data_int = data_int.loc[(data_int['mu'] == mu) & (data_int['theta'] == theta)][['x','y','u']] # To begin, I try with a solution from the training set
data_bc  = data_bc.loc [(data_bc ['mu'] == mu) & (data_bc ['theta'] == theta)][['x','y','u']] # 



# %% Plot numerical and NN solution

x_fit     = tf.constant(data_int[['x', 'y']])

u_num_fit = tf.constant(data_int[['u']])

fit_x_par = tf.stack((x_fit[:,0],x_fit[:,1],mu*np.ones(len(x_fit)),theta*np.ones(len(x_fit))),axis=1)

u_NN_int = pretrained_model(fit_x_par)

u_num_fit_normalized = (u_num_fit - min_train) / (max_train - min_train)



fig1 = plt.figure(dpi=300)

ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(xs = x_fit[:,0], ys = x_fit[:,1], zs = u_num_fit_normalized, label = 'Numerical solution (normalized)')
ax1.scatter(xs = x_fit[:,0], ys = x_fit[:,1], zs = u_NN_int,             label = 'Neural network solution')

ax1.legend()



# %% Utilities for loss on fit

def fit_loss(mu_curr,theta_curr): 

    fit_x_param = tf.stack((x_fit[:,0],x_fit[:,1],mu_curr*np.ones(len(x_fit)),theta_curr*np.ones(len(x_fit))),axis=1)     # Input for the neural network
    
    u_NN_fit = pretrained_model(fit_x_param)                                                                              # Output of the neural network

    return np.sum(np.power(u_num_fit_normalized - u_NN_fit, 2))



# %% Utilities for loss on bc

x_bc      = tf.constant(data_bc[['x', 'y']])

u_num_bc  = tf.constant(data_bc[['u']], ns.config.get_dtype())
u_num_bc_normalized = (u_num_bc - min_train) / (max_train - min_train)

def bc_loss (mu_curr,theta_curr): 

    bc_x_param = tf.stack((x_bc[:,0],x_bc[:,1],mu_curr*np.ones(len(x_bc)),theta_curr*np.ones(len(x_bc))),axis=1)

    u_NN_bc = pretrained_model(bc_x_param) 

    return np.sum(np.power(u_num_bc_normalized - u_NN_bc, 2))



# %% Utilities for loss on pde
    

def pde_loss(mu_curr, theta_curr):

    with ns.GradientTape(persistent = True) as tape:
        
        tape.watch(x_fit)

        # Domain, output and forcing

        pde_x_param = tf.stack((x_fit[:,0], x_fit[:,1], mu_curr*np.ones(len(x_fit)), theta_curr*np.ones(len(x_fit))), axis=1)

        u = pretrained_model(pde_x_param)

        f = forcing(pde_x_param)[:,None]

        # Gradient and laplacian

        grad_u = nse.physics.tens_style.gradient_scalar (tape, u, x_fit)
        lapl_u = nse.physics.tens_style.laplacian_scalar(tape, u, x_fit, domain_dim)[:,None]
        
        # Convective summand
        
        grad_u_per_b = (grad_u[:,0]*tf.math.cos(theta_curr) + grad_u[:,1]*tf.math.sin(theta_curr))[:,None]
        
        # Residual of the equation

        res = - mu_curr * (max_train - min_train) * lapl_u + 10 * (max_train - min_train) * grad_u_per_b - f
        
    return np.sum(np.power(res, 2))




# %% Compute and plot the losses on the space {mu, theta}

mus    = np.linspace(1 - 0.5, 5       + 0.5, 50)
thetas = np.linspace(0 -   1, 2*np.pi +   1, 50)

MU, THETA = np.meshgrid(mus, thetas)

LOSS_fit = np.zeros(MU.shape)
LOSS_pde = np.zeros(MU.shape)
LOSS_bc  = np.zeros(MU.shape)

for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        
        LOSS_fit[i,j] = fit_loss(MU[i,j], THETA[i,j])
        LOSS_pde[i,j] = pde_loss(MU[i,j], THETA[i,j])
        LOSS_bc [i,j] = bc_loss (MU[i,j], THETA[i,j])



fig2 = plt.figure(dpi=300)

ax2 = plt.axes(projection='3d')

ax2.plot_surface(MU, THETA, LOSS_fit, cmap=cm.jet)

ax2.set_title ('Fit loss')
ax2.set_xlabel('mu')
ax2.set_ylabel('theta')
ax2.set_zlabel('loss')



fig3 = plt.figure(dpi=300)

ax3 = plt.axes(projection='3d')

ax3.plot_surface(MU, THETA, LOSS_bc , cmap=cm.jet)

ax3.set_title ('BC loss')
ax3.set_xlabel('mu')
ax3.set_ylabel('theta')
ax3.set_zlabel('loss')



fig4 = plt.figure(dpi=300)

ax4 = plt.axes(projection='3d')

ax4.plot_surface(MU, THETA, LOSS_pde , cmap=cm.jet)

ax4.set_title ('PDE loss')
ax4.set_xlabel('mu')
ax4.set_ylabel('theta')
ax4.set_zlabel('loss')