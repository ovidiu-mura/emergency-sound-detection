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

class BNoise:
    def __init__(self):
        self.samples = None

    def brownian(self):
        # Process parameters
        delta = 1.25
        dt = 1.5

        # Initial condition.
        x = 0.0
        a = []
        # Number of iterations to compute.
        n = 48000

        # Iterate to compute the steps of the Brownian motion.
        for k in range(n*6):
            x = x + norm.rvs(scale=delta**2*dt)
            a.insert(k, x)
        self.samples = a
        return a

    def to_int32(self):
        # Take samples in [-1, 1] and scale to 16-bit integers,
        # values between -2^15 and 2^15 - 1.
        return int32(np.array(self.samples)*(2**20))

    def plot(self):
        spf = wave.open('test.wav','r')

        #Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, 'int32')
        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(signal[0:1600])
        plt.show()
