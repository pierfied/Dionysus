import numpy as np
from scipy.io import wavfile
from scipy import signal
from math import pi

class WavAnalysis:
    def __init__(self,fname,fft_len):
        self.r,self.d = wavfile.read(fname)

        self.ii16 = np.iinfo(np.int16)
        self.d = self.d.astype(np.float32)
        self.d /= self.ii16.max

        self.fft_len = fft_len

    def analyze(self):
        f, t, s = signal.stft(self.d, self.r, nperseg=self.fft_len)
        s = s.T

        mags = np.abs(s)
        # mags = mags[1:,:]

        mag_maxes = np.max(mags,0)
        self.mag_norm_coeffs = signal.savgol_filter(mag_maxes, 101, 2)

        norm_mags = np.copy(mags)
        for i in range(mags.shape[0]):
            norm_mags[i,:] /= self.mag_norm_coeffs

        phases = np.angle(s)
        # phase_diffs = np.diff(phases, axis=0)
        norm_phase_diffs = (phases + pi)/(2*pi)

        x = norm_mags * norm_phase_diffs
        y = norm_mags * (1 - norm_phase_diffs)

        z = np.concatenate((x,y),-1)

        return z

    def write_out(self,fname,data):
        x,y = np.split(data,2,-1)

        norm_phase_diffs = np.nan_to_num(x/(x+y))
        norm_mags = np.nan_to_num(x/norm_phase_diffs)

        mags = norm_mags * self.mag_norm_coeffs
        phases = (2*pi*norm_phase_diffs) - pi

        z = mags * np.exp(phases*1j)

        t,d = signal.istft(z.T,self.r,nperseg=self.fft_len)

        d = (d*self.ii16.max).astype(np.int16)

        wavfile.write(fname,self.r,d)