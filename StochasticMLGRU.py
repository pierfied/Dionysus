import tensorflow as tf
from tensorflow.contrib import rnn
from AutoScope import define_scope

class StochasticMLGRU:
    def __init__(self,num_layers,hidden_size,x=None,init_state=None):
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        with tf.name_scope('StochasticMLGRU'):
            if x is None:
                self.x = tf.placeholder(tf.float32)
            else:
                self.x = x

            self.batch_size = tf.shape(self.x)[0]
            self.seq_len = tf.shape(self.x)[1]
            self.feature_len = tf.shape(self.x)[2]

            if init_state is None:
                self.init_state = tf.placeholder(tf.float32)
            else:
                self.init_state = init_state

            self.predict_pdf
            self.sample
            self.loss

    @define_scope
    def predict_pdf(self):
        GRU = rnn.MultiRNNCell([rnn.GRUCell(self.hidden_size) for i in range(self.num_layers)])

        GRU_out,self.final_state = tf.nn.dynamic_rnn(GRU,self.x,initial_state=self.init_state)

        W_mu = tf.Variable(tf.truncated_normal([self.hidden_size,self.feature_len]))
        b_mu = tf.Variable(tf.truncated_normal([self.feature_len]))

        mu = tf.sigmoid(tf.matmul(GRU_out,W_mu) + b_mu)

        W_sigma = tf.Variable(tf.truncated_normal([self.hidden_size, self.feature_len]))
        b_sigma = tf.Variable(tf.truncated_normal([self.feature_len]))

        sigma = tf.exp(tf.matmul(GRU_out,W_sigma) + b_sigma)

        dists = tf.distributions.Normal(mu,sigma)

        return dists

    @define_scope
    def sample(self):
        return self.predict_pdf.sample()

    @define_scope
    def loss(self):
        probs = tf.clip_by_value(self.predict_pdf.prob(),1e-6,1e6)

        return tf.reduce_mean(-tf.log(probs))

    @define_scope
    def optimizer(self):
        return tf.train.AdamOptimizer(name='StochasticMLGRU_Adam').minimize(self.loss)