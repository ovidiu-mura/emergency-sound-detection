import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq


class READ_WAV_FFT:
    def __init__(self):
        self.samplerate = None
        self.data = None
        self.fftabs = None
        self.freqs = None
        self.file_name = None

    def read_wav_fft(self, file_name='eSound.wav'):
        self.file_name = file_name
        self.samplerate, self.data = wavfile.read(file_name)
        samples = self.data.shape[0]
        datafft = fft(self.data)

        # Get the absolute value of real and complex component
        self.fftabs = abs(datafft)
        self.freqs = fftfreq(samples,1/self.samplerate)


    def plot(self):
        plt.xlim( [10, self.samplerate/2] )
        plt.title('FFT - file: ' + str(self.file_name))
        plt.xscale( 'log' )
        plt.grid( True )
        plt.xlabel( 'Frequency (Hz)' )
        if('mix' in self.file_name):
            plt.plot(self.freqs[1500:int(self.freqs.size/2)-10000],self.fftabs[1500:int(self.freqs.size/2)-10000])
        else:
            plt.plot(self.freqs[:int(self.freqs.size/2)],self.fftabs[:int(self.freqs.size/2)])
        plt.show()

r = READ_WAV_FFT()
r.read_wav_fft()
r.plot()
