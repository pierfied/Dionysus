import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy import signal
from os.path import expanduser
from STFT import STFT
from Autoencoder import Autoencoder
from WavAnalysis import WavAnalysis

fft_len = 512
hidden_size = 256
code_len = 128
batch_size = 256

song = WavAnalysis(expanduser('~/Downloads/veldt.wav'),fft_len)
data = song.analyze()

x = tf.placeholder(tf.float32)
ae = Autoencoder(data.shape[1],hidden_size,code_len,x=data)
optimizer = ae.optimizer
loss = ae.loss
decoder = ae.decode

num_steps = 100001
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('/tmp/tf', sess.graph)

    for i in range(num_steps):
        batch_inds = np.random.choice(range(data.shape[0]),batch_size)
        batch = data[batch_inds,:]

        _,l = sess.run([optimizer,loss],feed_dict={x:batch})

        if i % 1000 == 0:
            print(i,' ',l)

        if i % 1000 == 0:
            recov = sess.run(decoder,feed_dict={x:data})

            fname = 'ae_%d.wav' % i
            song.write_out(fname,recov)