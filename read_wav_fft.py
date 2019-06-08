# author: Ovidiu Mura
# date: May 22, 2019

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq, ifft


# It provides the implementation of the READ_WAV_FFT class which reads the frequencies using FFT (Fast Fourier Transform)
# algorithm from a signal which can be read from file or passed as argument. The frequencies can be visualized using the
# plot_fft and plot methods.
class READ_WAV_FFT:
    def __init__(self):
        self.samplerate = None
        self.data = None
        self.fftabs = None
        self.freqs = None
        self.file_name = None

    # It finds the frequencies from a given signal stored in a file.
    def read_wav_fft(self, file_name='SineWave_440Hz.wav'):
        self.file_name = file_name
        self.samplerate, self.data = wavfile.read(file_name)
        samples = self.data.shape[0]
        datafft = fft(self.data)
        samples = self.data.shape[0]
        datafft = fft(self.data)

        # Get the absolute value of real and complex component
        self.fftabs = abs(datafft)
        self.freqs = fftfreq(samples,1/self.samplerate)

    # It finds the frequenceis of a given signal
    def read_fft(self, signal, samplerate):
        self.data = signal
        self.samplerate = samplerate
        datafft = fft(self.data)
        samples = len(self.data)
        datafft = fft(self.data)
        samples = len(self.data)

        # Get the absolute value of real and complex component
        self.fftabs = abs(datafft)
        self.freqs = fftfreq(samples,1/self.samplerate)

    # It plots the frequencies for visualization.
    def plot_fft(self):
        plt.xlim( [10, self.samplerate/2] )
        plt.title('FFT - file: ' + str(self.file_name))
        plt.xscale( 'log' )
        plt.grid( True )
        plt.xlabel( 'Frequency (Hz)' )
        plt.plot(self.freqs[0:int(self.freqs.size/2)],self.fftabs[0:int(self.freqs.size/2)])
        for i in range(int(self.freqs.size/2)):
            if(self.fftabs[i] > 1000000):
                idx = list(self.fftabs).index(self.fftabs[i])
                #print(self.freqs[idx])
        plt.show()

    # It plots the found frequencies
    def plot(self):
        plt.xlim( [10, self.samplerate/2] )
        plt.title('FFT - file: ' + str(self.file_name))
        plt.xscale( 'log' )
        plt.grid( True )
        plt.ylabel( 'Amplitudes' )
        plt.xlabel( 'Frequency (Hz)' )
        if('mix' in self.file_name):
            plt.plot(self.freqs[0:int(self.freqs.size/2)-10000],self.fftabs[0:int(self.freqs.size/2)-10000], color="blue")
        else:
            #plt.plot(self.freqs[:int(self.freqs.size/2)],self.fftabs[:int(self.freqs.size/2)], color="blue")
            plt.plot(self.freqs,self.fftabs, color="blue")
        serie = self.file_name + " Signal"
        plt.legend((serie, serie), loc="upper right")
        plt.show()
