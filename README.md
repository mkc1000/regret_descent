# Regret Descent
A decoration on any gradient-based optimization algorithm. Any parameters (such as learning rate, or momentum) are updated after each step toward what we might have wished those parameters to be for the last step (with Hessians assumed to be zero). Prototyped here in Tensorflow.

After each step, we calculate the gradient of the loss at the latest step with respect to each of the optimization parameters, while holding the location of the previous step constant. If the Hessian is implicated in such a calculation, we assume it is zero. Each optimization parameter has its own fixed learning rate, which we use to perform a simple gradient descent update on each optimization parameter after each step.

Here is a learning curve for NAG training a basic neural net for MNIST, alongside the same curve for regret-NAG.
![Learning Curves for NAG and Regret-NAG](https://github.com/mkc1000/regret_descent/blob/master/nag_regret_nag.png?raw=true)

*Details: Neural Net architecture--one ReLU activated hidden layer of width 10, cross-entropy loss function, minibatches of 100 for both training and loss calculation; NAG parameters--learning rate: 0.01, momentum: 0.9; Regret-NAG parameters--learning rate: 0.01, momentum: 0.9, learning rate learning rate: 0.01, momentum learning rate: 0.05*

Example Code:

    import tensorflow as tf
    from regret_descent import regret_descent_nag
    
    def test_loss(parameters):
        return tf.reduce_sum(tf.square(parameters))
    
    init_params = tf.Variable([1., 2., 3., 4., 5.])
    lr = tf.Variable(0.01)
    gamma = tf.Variable(0.9)
    lrlr = 0.001
    glr = 0.05
    opt_parameters, descent_history = regret_descent_nag(test_loss, init_params, lr, gamma, lrlr, glr, 8)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    loss_history = sess.run([descent_history['losses']])
    print loss_history

The code for algorithm is fairly straightforward. Here, for instance, is the definition of regret_descent_nag (17 lines excluding docstring):

    def regret_descent_nag(loss, init_parameters, learning_rate, gamma,
                           learning_rate_learning_rate, gamma_learning_rate,
                           n_steps):
        """Descend a loss surface using the Nesterov Accelerated Gradient Algorithm
        for `n_steps` steps, but adjusting the learning_rate and the value of gamma
        at every step.

        Arguments:
            loss:
                Function. loss maps a Tensorflow variable containing parameters to a
                Tensorflow variable representing the loss.
            init_parameters:
                Tensorflow variable. Inital parameters to start the descent from.
            learning_rate:
                Tensorflow variable (scalar shape). The initial learning rate for
                gradient descent steps.
            gamma:
                Tensorflow variable (scalar shape). The initial gamma (momentum) for
                NAG.
            learning_rate_learning_rate:
                Float. The learning rate for adjustments to the learning rate.
            gamma_learning_rate:
                Float. The learning rate for adjustments to gamma.
            n_steps:
                Int. the number of steps in the descent.

        Outputs:
            final_parameters:
                Tensorflow variable. Optimized parameters minimizing the loss function
            descent_history:
                Dictionary. Keys: ['parameters', 'losses', 'learning_rates',
                'gammas']. Values: lists of tensorflow variables
        """
        points = [init_parameters]
        losses = [loss(init_parameters)]
        updates = [tf.zeros_like(init_parameters)]
        learning_rates = [learning_rate]
        gammas = [gamma]
        for step in xrange(n_steps):
            gradient_calc_position = tf.stop_gradient(points[-1] - gammas[-1] * updates[-1])
            updates.append(updates[-1] * gammas[-1]
                + learning_rates[-1] * tf.gradients(loss(gradient_calc_position), gradient_calc_position)[0])
            points.append(points[-1] - updates[-1])
            losses.append(loss(points[-1]))
            learning_rate_gradient = tf.gradients(losses[-1], learning_rates[-1])[0]
            learning_rates.append(tf.reshape(learning_rates[-1]
                - learning_rate_gradient * learning_rate_learning_rate, shape=()))
            gamma_gradient = tf.gradients(losses[-1], gammas[-1])[0]
            gammas.append(tf.reshape(gammas[-1]
                - gamma_gradient * gamma_learning_rate, shape=()))
        descent_history = {'parameters' : points,
                           'losses' : losses,
                           'learning_rates' : learning_rates,
                           'gammas' : gammas}
        return points[-1], descent_history
