% Compare high-fidelity and deep-learning solutions 

x =     table2array(readtable('nodes.txt'));

u_hf =  table2array(readtable('high_fidelity_solution.txt'));
u_ann = table2array(readtable('neural_network_solution.txt'));

figure()
plot3(x(:,1),x(:,2), u_hf,  'bo')
hold on
grid on
plot3(x(:,1),x(:,2), u_ann, 'r.')
legend('High fidelity solution', 'Neural network solution')