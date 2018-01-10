import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from AutoScope import define_scope

class StochasticMLGRU:
    def __init__(self,num_layers,hidden_size,feature_len,x,y=None):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.feature_len = feature_len

        self.temp = tf.Variable(1.,False)

        with tf.name_scope('StochasticMLGRU'):
            self.x = x

            if y is None:
                self.y = tf.placeholder(tf.float32)
            else:
                self.y = y

            self.batch_size = tf.shape(self.x)[0]
            self.seq_len = tf.shape(self.x)[1]

            self.dists
            self.sample
            self.loss

    @define_scope
    def dists(self):
        self.GRU = rnn.MultiRNNCell([rnn.GRUCell(self.hidden_size) for i in range(self.num_layers)])

        self.init_state = self.GRU.zero_state(self.batch_size,tf.float32)

        GRU_out,self.final_state = tf.nn.dynamic_rnn(self.GRU,self.x,initial_state=self.init_state)

        GRU_out = tf.reshape(GRU_out,[-1,self.hidden_size])

        W = tf.Variable(tf.zeros([self.hidden_size,self.hidden_size]))
        b = tf.Variable(tf.zeros([self.hidden_size]))

        self.l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

        GRU_out = tf.nn.elu(tf.matmul(GRU_out,W) + b)

        W_mu = tf.Variable(tf.zeros([self.hidden_size,self.feature_len]))
        b_mu = tf.Variable(tf.zeros([self.feature_len]))

        self.l2_loss += tf.nn.l2_loss(W_mu) + tf.nn.l2_loss(b_mu)

        mu = tf.sigmoid(tf.matmul(GRU_out,W_mu) + b_mu)

        W_sigma = tf.Variable(tf.zeros([self.hidden_size, self.feature_len]))
        b_sigma = tf.Variable(tf.zeros([self.feature_len]))

        self.l2_loss += tf.nn.l2_loss(W_sigma) + tf.nn.l2_loss(b_sigma)

        sigma = tf.exp(tf.matmul(GRU_out,W_sigma) + b_sigma)

        mu = tf.reshape(mu,[self.batch_size,-1,self.feature_len])
        sigma = tf.clip_by_value(tf.reshape(sigma,[self.batch_size,-1,self.feature_len]),1e-6,
                                 1) * self.temp

        self.mu = mu
        self.sigma = sigma

        dists = tf.distributions.Normal(mu,sigma)

        return dists

    @define_scope
    def sample(self):
        return tf.clip_by_value(self.dists.sample(),0,1)

    @define_scope
    def loss(self):
        probs = tf.clip_by_value(self.dists.prob(self.y),1e-6,np.inf)
        loss = tf.reduce_mean(-tf.log(probs))

        # return tf.cond(tf.is_finite(loss),lambda: loss,lambda: 0.) + self.l2_loss
        return loss

    @define_scope
    def optimizer(self):
        return tf.train.AdamOptimizer(1e-4,name='StochasticMLGRU_Adam').minimize(self.loss)