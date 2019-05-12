# author: Ovidiu Mura
# date: April 30, 2019

# OSI licence: https://opensource.org/approval

from scipy.io.wavfile import write
from numpy import arange, pi, sin, int32, int16
import configparser
import numpy as np
import matplotlib.pyplot as plt
import wave

# from scikits import wavread

from noise import BNoise

class CSound:
    def __init__(self):
        None

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

    def plot_wave(self, file_name='slow.wav'):
        spf = wave.open(file_name,'r')

        #Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, 'int32')
        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(signal[0:1600])
        plt.show()

    def create_emergency_sound(self):
        N = 48000 # samples per second
        x = arange(0,3*N,0.5,float) # three seconds of audio
        data = self.f(x/N, 1000, 2, 100)
        write("slow.wav", N, self.to_int16(data))
        data = self.f(x/N, 1000, 8, 100)
        write("fast.wav", N, self.to_int16(data))
        self.plot_wave()


def main():
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    pn = config.get('DEFAULT','PROJECT_NAME')

    print(pn)
    exit(3)

    c = CSound()
    b = BNoise()
    b.brownian()
    cc = np.array(b.to_int32())
    print(cc)
    write('brown.wav', 48000, cc)
    b.plot()
    c.create_emergency_sound()
    c.plot_wave('fast.wav')

if __name__ == "__main__":
    main()
