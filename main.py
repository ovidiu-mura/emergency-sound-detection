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
from noise import *

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

        # Extract Raw Audio from Wav File
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

    w = wNoise()
    w.create_white_noise()
    w.plot()

    p = pNoise()
    p.create_pink_noise()
    p.plot()

    mix = Mix()
    output_mix_1 = config.get('MIXED_SIGNALS', 'eSOUND_bNOISE')[1:-1]
    output_mix_2 = config.get('MIXED_SIGNALS', 'eSOUND_wNOISE')[1:-1]
    output_mix_3 = config.get('MIXED_SIGNALS', 'eSOUND_pNOISE')[1:-1]

    mix.avg_mix_sounds('eSound.wav', 'bNoise.wav', output_mix_1)

    if(mix.is_in_mix('eSound.wav', 'bNoise.wav') == True):
        print("info: Emergency Sound found in the mix signal!")
        mix.plot_mix_and_original_signal()

    mix.avg_mix_sounds('eSound.wav', 'wNoise.wav', output_mix_2)

    if(mix.is_in_mix('eSound.wav', 'wNoise.wav') == True):
        print("info: Emergency Sound found in the mix signal!")
        mix.plot_mix_and_original_signal()

    mix.avg_mix_sounds('eSound.wav', 'pNoise.wav', output_mix_3)

    if(mix.is_in_mix('eSound.wav', 'pNoise.wav') == True):
        print("info: Emergency Sound found in the mix signal!")
        mix.plot_mix_and_original_signal()

    conv = Convolute()
    conv.convolve_gaussian_window(mix.samples_1, mix.samples_2)
    conv.fft_convolve(mix.samples_1, mix.samples_2)

    rwf = READ_WAV_FFT()
    rwf.read_wav_fft('eSound.wav')
    rwf.plot()

    rwf.read_wav_fft(output_mix_1)
    rwf.plot()

    rwf.read_wav_fft(output_mix_2)
    rwf.plot()

    rwf.read_wav_fft(output_mix_3)
    rwf.plot()

if __name__ == "__main__":
    main()
