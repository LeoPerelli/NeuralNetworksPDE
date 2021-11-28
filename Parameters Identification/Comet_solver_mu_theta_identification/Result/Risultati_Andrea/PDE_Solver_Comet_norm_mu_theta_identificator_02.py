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
import math



# %% Settings

domain_dim = 2
params_dim = 2

x_0 = np.array([0.5,0.5])

boundary  = lambda x: 0 * x[:,0]
forcing   = lambda x: 10 * np.exp(-100 * np.sqrt( np.power(x[:,0] - x_0[0], 2) + np.power(x[:,1] - x_0[1], 2) ))


mu_guess_0    = 5.
theta_guess_0 = 2*np.pi



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

mu_true    = data_int['mu'].unique()   [3]
theta_true = data_int['theta'].unique()[10]


data_int = data_int.loc[(data_int['mu'] == mu_true) & (data_int['theta'] == theta_true)][['x','y','u']] # To begin, I try with a solution from the training set
data_bc  = data_bc.loc [(data_bc ['mu'] == mu_true) & (data_bc ['theta'] == theta_true)][['x','y','u']] # 



# %% Variables initialization

mu    = ns.Variable(initial_value = mu_guess_0,    trainable = True, name = 'mu'   )
theta = ns.Variable(initial_value = theta_guess_0, trainable = True, name = 'theta')

variables = [mu, theta]



# %% Loss definitions

x_fit     = tf.constant(data_int[['x', 'y']])                            # Points on which we have evaluated the numerical solution

u_num_fit = tf.constant(data_int[['u']])                                 # Values of the numerical solution

u_num_fit_normalized = (u_num_fit - min_train) / (max_train - min_train) # We apply the same normalization we applied on the training data



# %% Loss on the residual of the PDE

# def PDE(): # we first tried to work with the fit loss

# loss_PDE = ns.LossMeanSquares(name = 'PDE', eval_roots = PDE, weight = 1)



# %% Loss on the fitting in the internal points

def fit_loss(): 
    
    fit_x_param = tf.stack((x_fit[:,0],x_fit[:,1],mu*np.ones(len(x_fit)),theta*np.ones(len(x_fit))),axis=1)     # Input for the neural network
    
    u_NN_fit = pretrained_model(fit_x_param)                                                                    # Output of the neural network
    
    return (u_num_fit_normalized - u_NN_fit)


loss_fit = ns.LossMeanSquares(name = 'fit', eval_roots = fit_loss, weight = 1)



# %% Losses gathering

losses = [loss_fit]



# %% Minimization

pltcb = ns.utils.HistoryPlotCallback()
pb    = ns.OptimizationProblem(variables, losses)

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate = 0.01),   num_epochs = 1000)
ns.minimize(pb, 'scipy', 'L-BFGS-B', options={'gtol': 1e-100},             num_epochs = 9000)


pltcb.finalize(pb, block = False)
print(variables)
