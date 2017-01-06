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
    lrlr = tf.Variable(0.001)
    glr = tf.Variable(0.05)
    opt_parameters, descent_history = regret_descent_nag(test_loss, init_params, lr, gamma, lrlr, glr, 8)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    loss_history = sess.run([descent_history['losses']])
    print loss_history
