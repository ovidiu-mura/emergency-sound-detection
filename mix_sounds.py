# Author: Ovidiu Mura
# Date: May 22, 2019

import wave
import numpy as np
from numpy import int16
import matplotlib.pyplot as plt

from scipy.io.wavfile import write
from correlate import *


class Mix:

    def __init__(self):
        self.file1 = None
        self.file2 = None
        self.orig_emergency = None
        self.mix_signal = None
        self.normalized_cross_correlation = None

    # convert samples from strings to ints
    def bin_to_int(self, bin):
        i = 0
        for c in bin[::-1]: # iterate over each char in reverse (because little-endian)
            # get the integer value of char and assign to the lowest byte of as_int, shifting the rest up
            i <<= 8
            i += c
        return i

    def mix_sounds(self, file1, file2):
        w1 = wave.open(file1)
        w2 = wave.open(file2)

        # get samples formatted as a string.
        samples1 = w1.readframes(w1.getnframes())
        samples2 = w2.readframes(w2.getnframes())

        # takes every 2 bytes and groups them together as 1 sample. ("123456" -> ["12", "34", "56"])
        samples1 = [samples1[i:i+2] for i in range(0, len(samples1), 2)]
        samples2 = [samples2[i:i+2] for i in range(0, len(samples2), 2)]

        samples1 = [self.bin_to_int(s) for s in samples1] #['\x04\x08'] -> [0x0804]
        samples2 = [self.bin_to_int(s) for s in samples2]
        self.orig_emergency = np.array(samples2)/10000

        # average the samples:
        samples_avg = [(s1+s2)/2 for (s1, s2) in zip(samples1, samples2)]
        self.mix_signal = samples_avg

        write("mix_brown_fast.wav", 48000, self.to_int16(samples_avg))
        return samples_avg

    def to_int16(self, signal):
        # Take samples in [-1, 1] and scale to 16-bit integers,
        # values between -2^15 and 2^15 - 1.
        return int16(signal*1)


    def is_in_mix(self):
        ys = self.mix_sounds("bNoise.wav", "eSound.wav") #[0:238]

        ys = np.absolute(ys)

        #from filters import *

        # a = np.array(filter_amps_below(ys, 1000))/10000
        #
        # ba = min(a[50:])
        # c = np.array(a)
        # c = np.array([(0 if x<0 else x) for x in c])/10000

        plt.figure(1)
        plt.title('Signal Wave...')

        self.mix_signal = ys/10000

        # x1 = mix_sig[0:1696]
        x1 = self.mix_signal
        # x2 = self.orig_emergency[0:1696]
        x2 = self.orig_emergency

        norm_corr = Correlate()
        self.normalized_cross_correlation = norm_corr.normalized_correlation(x1, x2)
        print("norm cross_corr: " + str(self.normalized_cross_correlation))
        # cor = norm_corr.similarity(mix_sig[0:1696], self.orig_emergency[0:1696])/10000

        print("std corr: " + str(norm_corr.standard_correlate(x1,x2)))
        if(self.normalized_cross_correlation > 0.5):
            return True
        return False

    def plot_mix_and_original_signal(self):
        plt.plot(self.mix_signal, color='blue')
        plt.plot(self.orig_emergency, color='red')
        # plt.plot(a, color='red')

        # plt.plot((cor), color='green')

        plt.show()
