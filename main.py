# author: Ovidiu Mura
# date: April 30, 2019

# OSI licence: https://opensource.org/approval

from scipy.io.wavfile import write
from numpy import arange, pi, sin, int32, int16
import configparser
import numpy as np
import matplotlib.pyplot as plt
import wave
from read_wav_fft import READ_WAV_FFT
from mix_sounds import *

# from scikits import wavread

from noise import bNoise

class eSound:
    def __init__(self):
        self.N = 48000
        self.file_name = 'eSound.wav'

    def f(self, t, f_c, f_m, beta):
        # t    = time
        # f_c  = carrier frequency
        # f_m  = modulation frequency
        # beta = modulation index
        return sin(2*pi*f_c*t - beta*sin(2*f_m*pi*t))

    def to_int16(self, signal):
        # Take samples in [-1, 1] and scale to 16-bit integers,
        # values between -2^15 and 2^15 - 1.
        return int16(signal*(2**15-1))

    def plot_wave(self, file_name='eSound.wav'):
        spf = wave.open(file_name,'r')

        #Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, 'int32')
        plt.figure(1)
        plt.title('Emergency Sound Wave...')
        plt.plot(signal[0:1600])
        plt.show()

    def create_emergency_sound(self):
        x = arange(0,3*self.N,0.5,float) # three seconds of audio
        data = self.f(x/self.N, 1000, 8, 100)
        write(self.file_name, self.N, self.to_int16(data))


def main():
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    project_name = config.get('DEFAULT','PROJECT_NAME')

    print("Welcome to " + str(project_name) + "!")
    c = eSound()
    c.create_emergency_sound()
    c.plot_wave()

    b = bNoise()
    b.brownian_motion()
    b.plot()

    mix_sounds('eSound.wav', 'bNoise.wav')

    rwf = READ_WAV_FFT()
    rwf.read_wav_fft('eSound.wav')
    rwf.plot()

    rwf.read_wav_fft('mix_brown_fast.wav')
    rwf.plot()


if __name__ == "__main__":
    main()
