# NeuralNetworksPDE

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>


## Overview
It is a well know result that Neural Networks can asimptotically approximate any continous real function.
Following a <a href="https://arxiv.org/abs/1910.03193">paper</a> authored by George Em Karniadakis et al, we explore the approximation properties 
of a Neural Network with respect to nonlinear Operators.

In particular, the case study is focused on operators which have as input the parameters of a Partial Differential Equation and the coordinates in which to evaluate the solution,
and as output the solution evaluated in the points.
This is particularly useful to overcome the intensive computations involved in simulating a PDE for a set of paramaeters. After training the network instead, obtaining the "learned" simulations with various sets of parameters is much faster.

