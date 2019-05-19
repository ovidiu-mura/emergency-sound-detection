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
import configparser

class bNoise:

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config/config.ini')
        self.samples = None
        self.delta = 1.25
        self.dt = 1.5
        self.x = 0.0
        self.a = []
        self.n = 48000
        self.file_name = self.config.get('NOISE','BROWN_FILE')

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

    def plot(self, file_name='bNoise.wav'):
        spf = wave.open(file_name,'r')

        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, 'int32')
        plt.figure(1)
        plt.title('Brownian Motion Wave...')
        plt.plot(signal[0:1600])
        plt.show()

class wNoise:

    def __init__(self):
        self.amplitude = 100000
        self.framerate = 48000
        self.duration = 6
        self.time = np.arange(self.duration*self.framerate)/self.framerate
        self.noise = None
        self.config = configparser.ConfigParser()
        self.config.read('config/config.ini')
        self.file_name = self.config.get('NOISE','WHITE_FILE')

    def create_white_noise(self):
        self.noise = np.random.uniform(-self.amplitude, self.amplitude, self.time.shape[0])
        write(self.file_name[1:-1], 48000, int32(self.noise*2**10))

    def plot(self):
        plt.figure(1)
        plt.title('White Noise Wave...')
        plt.plot(self.noise[0:250])
        plt.show()

# https://en.wikipedia.org/wiki/Pink_noise

class pNoise:
    def __init__(self):
        self.signal = None
        self.amplitude = 100
        self.fs = None
        self.duration = 3
        self.framerate = 48000
        self.time = np.arange(self.duration*2*self.framerate) / self.framerate
        self.config = configparser.ConfigParser()
        self.config.read('config/config.ini')
        self.file_name = self.config.get('NOISE','PINK_FILE')

    def make_wave(self):
        np.random.seed(20)
        self.signal = np.random.uniform(-self.amplitude, self.amplitude, self.time.shape[0])

        # parameters to generate the freqs
        n = self.signal.shape[0]
        d = 1/self.framerate

        # spectrum of the sound wave or real values array
        hs = np.fft.rfft(self.signal.real)
        fs = np.fft.rfftfreq(n, d)

        # pink filter: S(f) = 1/f^a, where 0 < a < 2
        denom = fs.real**(1/2)
        denom[0] = 1
        hs = hs.real/denom

        hs = np.absolute(hs)#**2
        #fs = np.fft.rfft(fs)

        write(self.file_name[1:-1], 48000, self.to_int32(hs))

        plt.figure(1)
        plt.title('Pink Noise Wave...')

        plt.plot(fs.real[10750:], hs.real[10750:], linewidth=2, color='r')
        plt.show()

    def to_int32(self, signal):
        # Take samples in [-1, 1] and scale to 32-bit integers,
        # values between -2^20 and 2^20.
        return int32(np.asarray(signal.real)*(2**20))

w = wNoise()
w.create_white_noise()
w.plot()

n = pNoise()
n.make_wave()
