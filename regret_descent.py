import numpy as np
import tensorflow as tf

def regret_descent_basic(loss, init_parameters, learning_rate, meta_learning_rate, n_steps):
    """Descend a loss surface for `n_steps` steps, adjusting the learning_rate
    in order for it to become more like the learning rate we wish we had had in
    the last step.

    Arguments:
        loss: Function. A function that takes a Tensorflow variable of parameters and returns the loss
        init_parameters: Tensorflow variable. Inital parameters to start the descent from
        learning_rate: Tensorflow variable (scalar shape). The initial learning rate for gradient descent steps
        meta_learning_rate: Float. the learning rate for adjustments to the learning rate
        n_steps: Int. the number of steps in the descent

    Outputs:
        final_parameters: Tensorflow variable. Optimized parameters minimizing the loss function
        descent_history: Dictionary. Keys: ['parameters', 'losses']. Values: lists of tensorflow variables
    """
    points = [init_parameters]
    losses = []
    gradients = []
    learning_rates = [learning_rate]
    for step in xrange(n_steps):
        losses.append(loss(points[-1]))
        gradients.append(tf.gradients(losses[-1], points[-1]))
        if step != 0:
            learning_rate_gradient = tf.gradients(losses[-1], learning_rates[-1])
            learning_rates.append(tf.reshape(learning_rates[-1] - learning_rate_gradient * meta_learning_rate, shape=()))
        next_point = points[-1] - gradients[-1] * learning_rates[-1]
        points.append(tf.reshape(next_point, [-1]))
    losses.append(loss(points[-1]))
    descent_history = {'parameters' : points,
                       'losses' : losses,
                       'learning_rates' : learning_rates}
    return points[-1], descent_history

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

# Basic tests

def test_loss(parameters):
    return tf.reduce_sum(tf.square(parameters))

def test_regret_descent():
    init_params = tf.Variable([1., 2., 3., 4., 5.])
    lr = tf.Variable(0.0)
    mlr = tf.Variable(0.0001)
    opt_parameters, descent_history = regret_descent(test_loss, init_params, lr, mlr, 8)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    return sess.run([opt_parameters, descent_history])

def test_regret_descent_nag():
    init_params = tf.Variable([1., 2., 3., 4., 5.])
    lr = tf.Variable(0.0)
    gamma = tf.Variable(0.9)
    lrlr = tf.Variable(0.001)
    glr = tf.Variable(0.05)
    opt_parameters, descent_history = regret_descent_nag(test_loss, init_params, lr, gamma, lrlr, glr, 8)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    return sess.run([opt_parameters, descent_history])

if __name__ == '__main__':
    opt_parameters, descent_history = test_regret_descent_nag()
    print descent_history
