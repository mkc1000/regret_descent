import numpy as np
import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from regret_descent import regret_descent_nag, test_loss
import cPickle as pickle
# import matplotlib.pyplot as plt

class MnistAccess(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                          help='Directory for storing input data')
        FLAGS, unparsed = parser.parse_known_args()
        self.mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

class MnistLoss(MnistAccess):
    def loss(self, parameters):
        x, y_ = self.mnist.train.next_batch(500)

        W0 = tf.reshape(tf.slice(parameters, [0], [7840]), [784, 10])
        b0 = tf.reshape(tf.slice(parameters, [7840], [10]), [10])
        h0_ = tf.matmul(x, W0) + b0
        h0 = tf.nn.relu(h0_)
        W1 = tf.reshape(tf.slice(parameters, [7850], [100]), [10, 10])
        b1 = tf.reshape(tf.slice(parameters, [7950], [10]), [10])
        y = tf.matmul(h0, W1) + b1

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        return cross_entropy

    def loss(self, parameters):
        x, y_ = self.mnist.train.next_batch(500)

        W0 = tf.reshape(tf.slice(parameters, [0], [7840]), [784, 10])
        b0 = tf.reshape(tf.slice(parameters, [7840], [10]), [10])
        h0_ = tf.matmul(x, W0) + b0
        h0 = tf.nn.relu(h0_)
        W1 = tf.reshape(tf.slice(parameters, [7850], [100]), [10, 10])
        b1 = tf.reshape(tf.slice(parameters, [7950], [10]), [10])
        y = tf.matmul(h0, W1) + b1

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        return cross_entropy

def test_descent(n_times, n_steps, lr, gamma, lrlr=tf.Variable(0.0), gammalr=tf.Variable(0.0), initialization_std=0.1):
    mnist_loss = MnistLoss()
    loss = mnist_loss.loss

    losses = []
    initial_parameters = tf.placeholder(tf.float32, [7960])
    _, descent_history = regret_descent_nag(loss, initial_parameters, lr, gamma, lrlr, gammalr, n_steps)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for i in xrange(n_times):
        initial_parameters_i = np.float32(np.random.normal(loc=0.0, scale=initialization_std, size=(7960,)))
        losses.append(sess.run(descent_history['losses'], feed_dict={initial_parameters: initial_parameters_i}))
        # print "Finished descent #", i+1

    losses = np.array(losses)
    losses = np.mean(losses, axis=0)
    return losses

if __name__ == '__main__':
    nag_losses = test_descent(20, 100, tf.Variable(0.01), tf.Variable(0.9), tf.Variable(0.0), tf.Variable(0.0))
    nag_regret_losses = test_descent(20, 100, tf.Variable(0.01), tf.Variable(0.9), tf.Variable(0.01), tf.Variable(0.05))
    print "NAG Losses (learning rate: 0.01; momentum: 0.9)"
    print nag_losses
    print "NAG Regret Losses (learning rate learning rate: 0.01; momentum learning rate: 0.05)"
    print nag_regret_losses
    # with open('descent_nag_20_avg_lr01.pkl', 'w') as f:
    #     pickle.dump(nag_losses, f)
