# Regret Descent
A decoration on any gradient-based optimization algorithm. Any parameters (such as learning rate, or momentum) are updated after each step toward what we might have wished those parameters to be for the last step (with Hessians assumed to be zero). Prototyped here in Tensorflow.

Here is a learning curve for NAG training a basic neural net for MNIST, alongside the same curve for regret-NAG.
