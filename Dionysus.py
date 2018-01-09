import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy import signal
from os.path import expanduser
from STFT import STFT
from Autoencoder import Autoencoder
from WavAnalysis import WavAnalysis
from StochasticMLGRU import StochasticMLGRU

fft_len = 512
hidden_size = 256
code_len = 128
batch_size = 256

rnn_batch_size = 16
rnn_seq_len = 16

song = WavAnalysis(expanduser('~/Downloads/hybrid.wav'),fft_len)
data = song.analyze()

x = tf.placeholder(tf.float32)
ae = Autoencoder(data.shape[1],hidden_size,code_len,x=x)
ae_optimizer = ae.optimizer
ae_loss = ae.loss
code = ae.encode
decoder = ae.decode

num_layers = 3

rnn_x = tf.placeholder(tf.float32,shape=[None,None,code_len])
rnn_y = tf.placeholder(tf.float32,shape=[None,None,code_len])
rnn = StochasticMLGRU(num_layers,hidden_size,code_len,rnn_x,rnn_y)
rnn_init_state = rnn.init_state
rnn_final_state = rnn.final_state
rnn_optimizer = rnn.optimizer
rnn_loss = rnn.loss
samples = rnn.sample

saver = tf.train.Saver()

sess = tf.Session()

writer = tf.summary.FileWriter('/tmp/tf', sess.graph)
ae_summary = tf.summary.scalar('ae_loss',ae_loss)
rnn_summary = tf.summary.scalar('rnn_loss',rnn_loss)

num_steps = 200001

tf.global_variables_initializer().run(session=sess)

def train_ae(num_steps):
    for i in range(num_steps):
        batch_inds = np.random.choice(range(data.shape[0]),batch_size)
        batch = data[batch_inds,:]

        _,l,ae_sum = sess.run([ae_optimizer, ae_loss, ae_summary], feed_dict={x:batch})

        if i % 1000 == 0:
            print(i,' ',l)

            writer.add_summary(ae_sum,i)
            writer.flush()

        if i % 100000 == 0:
            recov = sess.run(decoder,feed_dict={x:data})

            fname = 'ae_test/ae_%d.wav' % i
            song.write_out(fname,recov)

            saver.save(sess,'./model/model.ckpt')

def train_rnn(num_steps):
    codes = sess.run(ae.encode,feed_dict={x:data})
    num_codes = codes.shape[0]

    def gen_batch(ind):
        batch = np.zeros([rnn_batch_size,rnn_seq_len+1,code_len])

        chunk_len = num_codes // rnn_batch_size

        for i in range(rnn_batch_size):
            offset = i * chunk_len + ind * rnn_seq_len
            inds = np.arange(rnn_seq_len+1) + offset
            inds = np.take(np.arange(num_codes),inds,mode='wrap')

            batch[i,:,:] = codes[inds,:]

        return batch

    def gen_samp(fname,seed,num_samps):
        s = sess.run(rnn.GRU.zero_state(1,tf.float32))

        samp,s = sess.run([samples,rnn_final_state],feed_dict={rnn_x:seed[np.newaxis,:,:],
                                                                rnn_init_state:s})

        samps = np.zeros((num_samps,code_len))

        for i in range(num_samps):
            samp = samp[0,-1,:]

            samps[i,:] = samp

            samp,s = sess.run([samples,rnn_final_state],feed_dict={rnn_x:samp[np.newaxis,
                                                                              np.newaxis,:],
                                                                   rnn_init_state:s})

        samp_fft = sess.run(decoder,feed_dict={code:samps})

        song.write_out(fname,samp_fft)

    s = sess.run(rnn.GRU.zero_state(rnn_batch_size,tf.float32))
    for i in range(num_steps):
        batch = gen_batch(i)

        _,l,s,rnn_sum,a,b = sess.run([rnn_optimizer,rnn_loss,rnn_final_state,rnn_summary,rnn.mu,
                                      rnn.sigma],
                                     feed_dict={
            rnn_x:batch[:,:-1,:],rnn_y:batch[:,1:,:],rnn_init_state:s})

        if i % 100 == 0:
            print(i,' ',l)

            b = b[np.isfinite(b)]
            print(b.mean())
            print(b.min())
            print(b.max())

            writer.add_summary(rnn_sum,i)
            writer.flush()

        if i % 10000 == 0:
            fname = 'gen_test/samp_%d.wav' % i
            gen_samp(fname,codes[:128,:],128)

            # saver.save(sess, './model/model.ckpt')

# saver.restore(sess,'./model/model.ckpt')

train_ae(200001)
train_rnn(500001)

writer.close()