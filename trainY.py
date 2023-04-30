# File for training y's
import tensorflow as tf

def trainY():

    x = tf.placeholder(name='x',shape=(None,1),dtype=tf.float32)
    layer = x
    for _ in range(3):
        layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.tanh)
    mu = tf.layers.dense(inputs=layer, units=1)
    sigma = tf.layers.dense(inputs=layer, units=1,activation=lambda x: tf.nn.elu(x) + 1)
    dist = tf.distributions.Normal(loc=mu, scale=sigma)
    loss = tf.reduce_mean(-dist.log_prob(y))