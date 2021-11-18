# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 15:53:38 2021

@author: carlo
"""


import nisaba as ns
import tensorflow as tf
import nisaba.experimental as nse
import numpy as np
import matplotlib.pyplot as plt

########################################################################################
#PROBLEM
#omega = (0, 2*pi)x(0, 2*pi)
# - mu * delta(u) + b * nabla(u) + sigma * u = f
# BC see from u_exact

# f = cos(y*x^2)(-4y+4xy-x^2) + sin(y*x^2)(2x^4+8x^2y^2+3)
# mu = 2
# b = [2; -1]
# sig = 3
# u_exact(x,y) = sin(y*x^2)


########################################################################################
#PROBLEM SETUP
data_type = ns.config.get_dtype()
W1 = 1.5
W2 = 1.5

dim = 2

mu_exact = 2
mu_guess = 0

b_exact = tf.constant([2, -1], shape=(2,), dtype = data_type)
b_exact = tf.reshape(b_exact, (2,1))
b_guess = tf.Variable([[0],[0]], dtype=data_type)


sig_exact = 3
sig_guess = 0

u_exact = lambda x: np.sin( (x[:,0] ** 2) * x[:,1] )
forcing = lambda x: np.cos((x[:,0]**2)*x[:,1])*(- 4*x[:,1] + 4*x[:,0]*x[:,1] - x[:,0]**2) + np.sin((x[:,0]**2)*x[:,1])*(2*(x[:,0]**4) + 8*(x[:,0]**2)*(x[:,1]**2)+3)


#numerical options
num_PDE = 1000      #points of numerical sol
num_BC = 50        #points of numerical BC
num_fit = 100      #points of fit (the training data)
num_test = 1000    #points of test

########################################################################################
#INITIALIZE THE NEURAL NETWORK

#set the seed
np.random.seed(1)
tf.random.set_seed(1)

#I create the NN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=20, input_shape=(2,), activation = 'tanh'))
model.add(tf.keras.layers.Dense(units=20, activation = 'tanh'))
model.add(tf.keras.layers.Dense(units=20, activation = 'tanh'))
#model.add(tf.keras.layers.Dense(units=20, activation = 'tanh'))
model.add(tf.keras.layers.Dense(units=1)) 

print(model.summary())

#I set the parameters as variables
mu = ns.Variable(mu_guess, name='mu')
b = ns.Variable(b_guess, name='b')
sig = ns.Variable(sig_guess, name='sig')

#########################################################################################
#INITIALIZE THE POINTS IN THE GRID

#grid of the numerical sol
x_PDE = tf.random.uniform(shape = [num_PDE, 2], minval = [-W1, -W2], maxval = [W1, W2], dtype=data_type)

#grid of the fitting sol (training data)
x_fit = tf.random.uniform(shape = [num_fit, 2], minval = [-W1, -W2], maxval = [W1, W2], dtype=data_type)

#grid of the test sol
x_test = tf.random.uniform(shape = [num_test, 2], minval = [-W1, -W2], maxval = [W1, W2], dtype=data_type)

#Boundary grid
#random points in layer ([0,0],[0,2pi])  (have two coords)
x_BC_x0 = tf.random.uniform(shape = [num_BC,2], minval = [-W1, -W2], maxval = [-W1, W2], dtype=data_type)
#random points in layer ([2pi,0],[2pi,2pi])  (have two coords)
x_BC_x1 = tf.random.uniform(shape = [num_BC,2], minval = [W1, -W2], maxval = [W1, W2], dtype=data_type)
#random points in layer ([0,0],[2pi,0])  (have two coords)
x_BC_y0 = tf.random.uniform(shape = [num_BC,2], minval = [-W1, -W2], maxval = [W1, -W2], dtype=data_type)
#random points in layer ([0,2pi],[2pi,2pi])  (have two coords)
x_BC_y1 = tf.random.uniform(shape = [num_BC,2], minval = [-W1, W2], maxval = [W1, W2], dtype=data_type)
#I assemble the BC in a unique (n,2) matrix
x_BC = tf.concat([x_BC_x0, x_BC_x1, x_BC_y0, x_BC_y1], axis=0)

#I valuate the grids in the exact sol
u_PDE = u_exact(x_PDE)
u_fit = u_exact(x_fit)
u_test = u_exact(x_test)
u_BC = u_exact(x_BC)    #it will be my Dirichlet BC

#########################################################################################
#PDE OPERATOR FOR THE RESIDUAL COMPUTATION
def PDE(x, u_model):
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x)
        u = u_model(x)
        u = tf.squeeze(u)    #remove unseful dimentions
        f = forcing(x)
        f = tf.squeeze(f)
        grad = nse.physics.tens_style.gradient_scalar(tape, u, x)
        lapl = nse.physics.tens_style.laplacian_scalar(tape, u, x, dim)
    return - mu * lapl + tf.squeeze(tf.matmul(grad,b)) + sig * u - f
    
########################################################################################
#LOSS DEFINITION
#normalization factor for the fitting
u_norm = tf.reduce_mean(tf.square(u_fit))    

losses = [
          ns.LossMeanSquares('fit', lambda: tf.squeeze(model(x_fit)) - u_fit, normalization = u_norm, weight=1),   #fit loss, MSE(output of NN - exact sol)
          ns.LossMeanSquares('PDE', lambda: PDE(x_PDE, model), weight=10),                                          #PDE residual loss
          ns.LossMeanSquares('BC', lambda: tf.squeeze(model(x_BC)) - u_BC, weight=0.1)
          ]                                                                                               #BC residual loss

#loss for validation
loss_test = [ns.LossMeanSquares('fit', lambda: tf.squeeze(model(x_test)) - u_test, normalization = u_norm),
             ns.LossMeanSquares('mu_error', lambda: (mu - mu_exact)/abs(mu_exact)), 
             ns.LossMeanSquares('b_error', lambda: tf.math.reduce_euclidean_norm(b-b_exact)/tf.math.reduce_euclidean_norm(b_exact)), 
             ns.LossMeanSquares('sig_error', lambda: (sig - sig_exact)/abs(sig_exact)),
             ns.LossMeanSquares('PDE_err', lambda: PDE(x_test, model), weight=1)
             ]

########################################################################################
#MINIMIZER AND TRAINING

#create the problem
pb = ns.OptimizationProblem(model.variables + [mu] + [b] + [sig], losses, loss_test)

#I train the model with NN of keraswit Adam optimizer
#ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs=3000)
#I continue the same training with a different optimizer
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 3000)

#ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs=500)

#I plot the losses during training
ns.utils.plot_history(pb)  

#I print the found values
print('mu (exact)     : %f' % mu_exact)
print('mu (identified): %f' % mu.numpy())
print(b_exact.numpy())
print(b.numpy())
print('sig (exact)     : %f' % sig_exact)
print('sig (identified): %f' % sig.numpy())

rel_err = np.array([abs(mu - mu_exact)/abs(mu_exact), 
                    tf.math.reduce_euclidean_norm(b-b_exact)/tf.math.reduce_euclidean_norm(b_exact),
                    abs(sig - sig_exact)/abs(sig_exact)])
print(rel_err)
#######################################################################################
#PLOT THE SOLUTION
from mpl_toolkits.mplot3d import Axes3D
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(x_test[:,0], x_test[:,1], u_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test).numpy(), label = 'num sol')
ax.legend()

#  %matplotlib qt
#  %matplotlib inline