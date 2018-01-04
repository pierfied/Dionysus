import tensorflow as tf
from AutoScope import define_scope

class Autoencoder:
    def __init__(self, data_len, hidden_size, code_len, x=None, code=None, name='autoencoder'):
        self.data_len = data_len
        self.hidden_size = hidden_size
        self.code_len = code
        self.name = name

        with tf.name_scope(name):
            if x is None:
                self.x = tf.placeholder(tf.float32)
            else:
                self.x = x

            if code is None:
                self.encode = Encoder(data_len, hidden_size, code_len, x).encode
            else:
                self.encode = code

            self.decode = Decoder(data_len, hidden_size, code_len, self.encode).decode

            self.loss
            self.optimizer

    @define_scope
    def loss(self):
        return tf.losses.mean_squared_error(self.x, self.decode)

    @define_scope
    def optimizer(self):
        return tf.train.AdamOptimizer(name=self.name+'_Adam').minimize(self.loss)

class Encoder:
    def __init__(self, data_len, hidden_size, code_len, x):
        self.data_len = data_len
        self.hidden_size = hidden_size
        self.code_len = code_len

        self.x = x

        self.encode

    @define_scope
    def encode(self):
        W1 = tf.Variable(tf.truncated_normal([self.data_len, self.hidden_size]))
        b1 = tf.Variable(tf.truncated_normal([self.hidden_size]))

        y1 = tf.nn.relu(tf.matmul(self.x,W1) + b1)

        W2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.code_len]))
        b2 = tf.Variable(tf.truncated_normal([self.code_len]))

        code = tf.matmul(y1,W2) + b2

        return code


class Decoder:
    def __init__(self, data_len, hidden_size, code_len, code):
        self.data_len = data_len
        self.hidden_size = hidden_size
        self.code_len = code_len

        self.code = code

        self.decode

    @define_scope
    def decode(self):
        W1 = tf.Variable(tf.truncated_normal([self.code_len, self.hidden_size]))
        b1 = tf.Variable(tf.truncated_normal([self.hidden_size]))

        y1 = tf.nn.relu(tf.matmul(self.code, W1) + b1)

        W2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.data_len]))
        b2 = tf.Variable(tf.truncated_normal([self.data_len]))

        code = tf.matmul(y1, W2) + b2

        return code