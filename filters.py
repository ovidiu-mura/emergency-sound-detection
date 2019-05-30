# author: Ovidiu Mura
# date: May 22, 2019

def filter_amps_below(signal, amp):
    # signal: signal to be filtered
    # amp: the threshold amplitude; the amps below this amp will be considered
    a = []
    for i in range(1,238):
        if(i<237):
            b=0
            if(len(a)!=0):
                b = a[-1]
            if(signal[i]<b+amp):
                a.insert(i, signal[i])
    a[0] = 0
    return a


def bin_to_int(bin):
    i = 0
    # iterate over each char in reverse (because little-endian)
    for c in bin[::-1]:
        # get the integer value of char and
        # assign to the lowest byte of as_int, shifting the rest up
        i <<= 8
        i += c
    return i

# https://plot.ly/python/fft-filters/

# TODO: low-pass, high-pass, band-pass filters

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# t = np.linspace(-1, 1, 201)
# x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1) + 0.18*np.cos(2*np.pi*3.85*t))
# xn = x + np.random.randn(len(t)) * 0.08

import wave
from scipy.io import wavfile
from scipy import signal

class Filter:

    def __init__(self):
        self.signal = None

    # https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
    def low_pass(self):
        samplerate, data = wavfile.read('eSound.wav')
        ts = np.arange(len(data))/float(samplerate)
        plt.plot(ts[:2000], data[:2000], label='orig')

        fs = samplerate
        fc = 180 # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(3, w, 'low')

        output = signal.filtfilt(b, a, data)
        plt.plot(ts[:2000], output[:2000], label='filtered')
        plt.legend()
        plt.grid(True)
        plt.show()

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    def lowpass(self):
        samplerate, xn = wavfile.read("SineWave_440Hz.wav")
        xn = np.array(xn)
        t = np.arange(len(xn))/float(samplerate)
        b, a = signal.butter(3, 0.015)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        y = signal.filtfilt(b, a, xn)

        # plt.figure
        plt.plot(t[:100], xn[:100], 'b') #, alpha=0.75)
        plt.plot(t[:100], z[:100], 'r--', t[:100], z2[:100], 'r', t[:100], y[:100], 'k')
        # plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice','filtfilt'), loc='best')
        plt.grid(True)
        plt.show()

    # https://www.programcreek.com/python/example/59508/scipy.signal.butter
    def highpass(self):
        cut_off = 500.0
        samplerate, xn = wavfile.read("SineWave_440Hz.wav")
        Wn = (float(cut_off) / (float(samplerate) / 2.0), 0.95)
        b, a = signal.butter(3, Wn, 'pass')
        z = signal.filtfilt(b, a, xn)
        plt.plot(xn[:100], 'k--')
        plt.plot(z[:100], 'r')
        plt.grid(True)
        plt.show()

    # https://github.com/neurotechuoft/Wall-EEG/blob/master/Code/OpenBCIPy/src/butter_tingz.py
    def bandpass(self):
        samplerate, xn = wavfile.read("SineWave_440Hz.wav")
        nyq = 0.5 * 250
        lowcut = 10
        highcut = 30
        low = lowcut / nyq
        high = highcut / nyq
        print(low)
        print(high)
        b, a = signal.butter(3, [low, high], btype='band')
        filtered_data = signal.lfilter(b, a, xn)

        plt.plot(xn[:100], 'k--')
        plt.plot(filtered_data[:100], 'r')
        plt.grid(True)
        plt.show()

f = Filter()
f.bandpass()
