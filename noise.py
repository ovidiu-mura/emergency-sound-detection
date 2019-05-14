# author: Ovidiu Mura
# date: May 1, 2019

# Formula of brown noise
# X(0)=X0
# X(t+dt)=X(t)+N(0,(delta)^2dt;t,t+dt)
# url: https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html

from scipy.stats import norm
from numpy import arange, pi, sin, int32
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

class BNoise:
    def __init__(self):
        self.samples = None
        self.delta = 1.25
        self.dt = 1.5
        self.x = 0.0
        self.a = []
        self.n = 48000
        self.file_name = 'brown.wav'

    def brownian_motion(self):
        for k in range(self.n*6):
            self.x = self.x + norm.rvs(scale=self.delta**2*self.dt)
            self.a.insert(k, self.x)
        self.samples = self.a
        cc = np.array(self.to_int32())
        write(self.file_name, 48000, cc)
        return self.a

    def to_int32(self):
        # Take samples in [-1, 1] and scale to 32-bit integers,
        # values between -2^20 and 2^20.
        return int32(np.array(self.samples)*(2**20))

    def plot(self, file_name='brown.wav'):
        spf = wave.open(file_name,'r')

        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, 'int32')
        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(signal[0:1600])
        plt.show()
