import tensorflow as tf
from AutoScope import define_scope
from tensorflow.contrib import signal

class STFT:
    def __init__(self,fft_len,sig=None,ft=None):
        self.fft_len = fft_len

        if sig is None:
            self.sig = tf.placeholder(tf.float32)
        else:
            self.sig = sig

        if ft is None:
            self.ft = tf.placeholder(tf.complex64)
        else:
            self.ft = ft

        self.stft
        self.istft

    @define_scope
    def stft(self):
        return signal.stft(self.sig,self.fft_len,self.fft_len//2,window_fn=signal.hamming_window)
    
    @define_scope
    def istft(self):
        return signal.inverse_stft(self.ft,self.fft_len,self.fft_len//2,
                                   window_fn=signal.hamming_window)