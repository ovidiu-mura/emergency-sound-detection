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

    # https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
    def low_pass(self):
        samplerate, data = wavfile.read('SineWave_440Hz.wav')
        ts = np.arange(len(data))/float(samplerate)
        #plt.plot(ts[:2000], data[:2000], label='orig')
        x = rfft(data)
        x = np.abs(x)
        fs = samplerate
        fc = 130.0 # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(8, w, 'low')

        output = signal.filtfilt(b, a, x)

        y = irfft(output)
        #y = np.abs(y)
        f = rfftfreq(len(y), 1/fs)

        #plt.plot(f[:int(f.size/2)-58000], y[:int(y.size/2)-58000], 'r')
        plt.plot(f[0:5000], y[0:5000], 'b')
        plt.plot(f[0:5000], data[0:5000], 'r')
        plt.grid(True)
        plt.show()

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    def lowpass(self):
        samplerate, x0 = wavfile.read("SineWave_440Hz.wav")
        xn = np.array(x0)
        xn = fft(xn)
        t = np.arange(len(xn))/float(samplerate)
        cut_off = 500.0
        Wn = (float(cut_off) / (float(samplerate) / 2.0))
        b, a = signal.butter(3, 0.25, 'low')
        # zi = signal.lfilter_zi(b, a)
        # z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
        # z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        y = signal.filtfilt(b, a, xn)

        yy = ifft(y)
        xx = fftfreq(len(xn), 1/samplerate)

        plt.plot(xx[2000:25000].real, x0[2000:25000].real, 'red')
        plt.plot(xx[2000:25000].real, yy[2000:25000].real, 'green')
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


from read_wav_fft import *


f = Filter()

f.bandpass()
