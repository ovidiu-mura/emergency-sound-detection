import wave
import numpy as np
from numpy import int16
import matplotlib.pyplot as plt

from scipy.io.wavfile import write


# convert samples from strings to ints
def bin_to_int(bin):
    as_int = 0
    for char in bin[::-1]: #iterate over each char in reverse (because little-endian)
        #get the integer value of char and assign to the lowest byte of as_int, shifting the rest up
        as_int <<= 8
        as_int += char
    return as_int


def mix_sounds(file1, file2):
    w1 = wave.open(file1)
    w2 = wave.open(file2)

    # get samples formatted as a string.
    samples1 = w1.readframes(w1.getnframes())
    samples2 = w2.readframes(w2.getnframes())

    # takes every 2 bytes and groups them together as 1 sample. ("123456" -> ["12", "34", "56"])
    samples1 = [samples1[i:i+2] for i in range(0, len(samples1), 2)]
    samples2 = [samples2[i:i+2] for i in range(0, len(samples2), 2)]

    samples1 = [bin_to_int(s) for s in samples1] #['\x04\x08'] -> [0x0804]
    samples2 = [bin_to_int(s) for s in samples2]

    # average the samples:
    samples_avg = [(s1+s2)/2 for (s1, s2) in zip(samples1, samples2)]
    write("mix_brown_fast.wav", 48000, to_int16(samples_avg))
    return samples_avg


def to_int16(signal):
    # Take samples in [-1, 1] and scale to 16-bit integers,
    # values between -2^15 and 2^15 - 1.
    return int16(signal*1)


plt.figure(1)
plt.title('Signal Wave...')
plt.plot(mix_sounds("bNoise.wav", "eSound.wav")[0:1000])
plt.show()
