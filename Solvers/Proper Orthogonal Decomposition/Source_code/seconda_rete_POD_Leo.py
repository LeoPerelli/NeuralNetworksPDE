
#Da aggiungere in fondo allo script precedente del POD

#I create the 2nd model
model_coeffs = tf.keras.Sequential([
                                tf.keras.layers.Dense(24, input_shape=(params_dim,), activation=tf.nn.tanh),
                                tf.keras.layers.Dense(24, activation=tf.nn.tanh),
                                tf.keras.layers.Dense(K+1)
                            ])



# Split in training and validation data
n_train = int(train_split*p_all.shape[0])
total_n = p_all.shape[0]

idx = [i for i in range(total_n)]

idx_train = np.sort(np.array(random.sample(idx, n_train)))
idx_test  = np.sort(np.setdiff1d(idx, idx_train))


p_train = tf.gather(p_all,      idx_train)
p_test  = tf.gather(p_all,      idx_test)

u_train = tf.gather(S,      idx_train)
u_test  = tf.gather(S,      idx_test)




#I collect the estimate of the phi_i from previous model:
phi_i_and_u_0_trained = model(x_int_tf)

# %% Losses definition
def train_loss():

    # somma di phi_i * coeff_i + u_0_i + b
    return tf.linalg.matmul(model_coeffs(p_train)[:,1:K+1],phi_i_and_u_0_trained[:,1:K+1],transpose_b = True) + tf.tile(tf.transpose(tf.expand_dims(phi_i_and_u_0_trained[:,0], axis=1)), [len(idx_train), 1]) + tf.tile(tf.expand_dims(model_coeffs(p_train)[:,0],axis = 1),[1,x_int_tf.shape[0]]) - u_train

def test_loss():
    # somma di phi_i * coeff_i + u_0_i + b
    return tf.linalg.matmul(model_coeffs(p_test)[:,1:K+1],phi_i_and_u_0_trained[:,1:K+1],transpose_b = True) + tf.tile(tf.transpose(tf.expand_dims(phi_i_and_u_0_trained[:,0], axis=1)), [len(idx_test), 1]) + tf.tile(tf.expand_dims(model_coeffs(p_test)[:,0],axis = 1),[1,x_int_tf.shape[0]]) - u_test

losses = [ns.LossMeanSquares(name = 'train', eval_roots = train_loss, weight = 1)]
losses_test = [ns.LossMeanSquares(name = 'test', eval_roots = test_loss, weight = 1)]



# %% Training

pb = ns.OptimizationProblem(model_coeffs.variables, losses, losses_test)


ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-3), num_epochs =  1000)
ns.minimize(pb, 'scipy', 'BFGS',                                       num_epochs = 2000, options={'gtol': 1e-100})


ns.utils.plot_history(pb)


#Plotto la soluzione della rete neurale per la prima coppia di parametri e anche quella numerica
fig5 = plt.figure(dpi=300)

ax5  = fig5.add_subplot(111, projection='3d')


ax5.scatter(xs = x_int[:,0], ys = x_int[:,1], zs = u_test[0,:], label = 'True solution')
ax5.scatter(xs = x_int[:,0], ys = x_int[:,1], zs = tf.linalg.matmul(model_coeffs(tf.expand_dims(p_test[0,:],axis = 0))[:,1:K+1],phi_i_and_u_0_trained[:,1:K+1],transpose_b = True) + tf.transpose(tf.expand_dims(phi_i_and_u_0_trained[:,0], axis=1))+ tf.expand_dims(model_coeffs(tf.expand_dims(p_test[0,:],axis = 0))[:,0],axis = 1), label = 'Trained Solution')

ax5.legend()
