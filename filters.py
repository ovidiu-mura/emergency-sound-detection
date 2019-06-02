# author: Ovidiu Mura
# date: May 22, 2019

# Reference: http://chris35wills.github.io/simple_fft_filter/

def filter_amps_below(signal, amp_below):
    # signal: signal to be filtered
    # amp: the threshold amplitude; the amps below this amp will be considered
    a = []
    for i in range(1,238):
        if(i<237):
            b=0
            if(len(a)!=0):
                b = a[-1]
            if(signal[i]<b+amp_below):
                a.insert(i, signal[i])
    a[0] = 0
    return a

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
from scipy.fftpack import fft,fftfreq, ifft, rfft, rfftfreq, irfft

class Filter:

    def __init__(self):
        self.signal = None
        self.goertzel_freqs = {}
        self.norm = None
        self.coeffs = None

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    # https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
    def lowpass(self):
        samplerate, x0 = wavfile.read("SineWave_440Hz.wav")
        #xn = np.array(x0)
        xn = fft(x0)
        t = np.arange(len(xn))/float(samplerate)
        cut_off = 500.0
        Wn = (float(cut_off) / (float(samplerate) / 2.0))

        b, a = signal.butter(3, Wn, 'low')

        y = signal.filtfilt(b, a, xn)

        yy = ifft(y)
        xx = fftfreq(len(xn), 1/samplerate)

        plt.plot(xx[:10000].real, x0[:10000].real, 'red')
        plt.plot(xx[:10000].real, yy[:10000].real, 'green')
        # plt.plot(xx[:int(xx.size/2)].real, yy[:int(yy.size/2)].real, 'green')

        # plt.figure
        # plt.plot(t[:100], xn[:100], 'b') #, alpha=0.75)
        # plt.plot(t[:100], y[:100], 'r') #, alpha=0.75)
        # plt.plot(t[:100], z[:100], 'r--', t[:100], z2[:100], 'r', t[:100], y[:100], 'k')
        # plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice','filtfilt'), loc='best')
        plt.grid(True)
        plt.show()

        # rwf = READ_WAV_FFT()
        # rwf.read_fft(y, samplerate)
        # rwf.plot_fft()

    # https://www.programcreek.com/python/example/59508/scipy.signal.butter
    def highpass(self):
        cut_off = 500.0
        samplerate, xn = wavfile.read("SineWave_440Hz.wav")

        f1 = fft(xn)

        Wn = (float(cut_off) / (float(samplerate) / 2.0))

        b, a = signal.butter(3, Wn, 'high')

        z = signal.filtfilt(b, a, f1)

        f2 = ifft(z)
        xx = fftfreq(len(f2), 1/samplerate)

        #plt.plot(xx[:int(xx.size/2)], f2[:int(f1.size/2)], 'r')
        plt.plot(xx[:10000].real, xn[:10000].real, 'r')
        plt.plot(xx[:10000].real, f2[:10000].real, 'green')
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

        x = fft(xn)

        b, a = signal.butter(3, [low, high], btype='band')

        filtered_data = signal.lfilter(b, a, x)

        #y = signal.filtfilt(b, a, )

        y = ifft(filtered_data)
        xx = fftfreq(len(xn), 1/samplerate)

        #plt.plot(xn[:100], 'k--')
        plt.plot(xx[:30000].real, xn[:30000].real, 'green')
        plt.plot(xx[:30000].real, y[:30000].real, 'r')
        plt.grid(True)
        plt.show()


    def goertzel(self, x, f):
        """x is an array of samples, f is the target frequency.
        Returns the output magnitude."""
        import math
        w0 = float((2*math.pi*f)/44100)
        n = len(x)
        self.norm = np.exp(1j * w0 * n)
        self.coeffs = np.exp(np.array([-1j * w0 * k for k in range(n)]))
        y = np.abs(self.norm * np.dot(self.coeffs, x))
        self.goertzel_freqs[f] = y
        return y


def get_sine_freq(p=True):
    f = Filter()

    samplerate, data = wavfile.read('SineWave_440Hz.wav')

    freqs = {}

    for i in range(150):
        m = f.goertzel(data, 420+i)
        freqs[420+i] = m

    m = max(freqs.values())
    idx = list(freqs.values()).index(m)
    freq = list(freqs.keys())[idx]

    if(p):
        plt.plot(freqs.keys(), freqs.values(), color='green')
        plt.show()

    return freq

# f = Filter()
# f.lowpass()
