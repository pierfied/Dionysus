import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from os.path import expanduser
from STFT import STFT
from Autoencoder import Autoencoder

r,d = wavfile.read(expanduser('~/Downloads/veldt.wav'))

fft_len = 1024
fft_out_len = fft_len//2+1
hidden_size = 256
code_len = 128

raw_sig = tf.placeholder(tf.float32)
stft = STFT(fft_len,raw_sig)
mags = tf.abs(stft.stft)
ae = Autoencoder(fft_out_len,hidden_size,code_len,mags,name='ae1')
code = ae.encode
ae2 = Autoencoder(fft_out_len,hidden_size,code_len,code=code,name='ae2')
out = ae2.decode

d = d[:fft_len*(len(d)//fft_len)]

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('/tmp/tf', sess.graph)

    c,o = sess.run([code,out],feed_dict={raw_sig:d})

    print(c.shape)

    writer.close()