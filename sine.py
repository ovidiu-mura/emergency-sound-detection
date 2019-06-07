# generate wav file containing sine waves
# FB36 - 20120617
import math, wave, array
import matplotlib.pyplot as plt
from scipy.io import wavfile

class Sine:
    def __init__(self):
        self.duration = 3 # seconds
        self.freq = 440 # of cycles per second (Hz) (frequency of the sine waves)
        self.volume = 100 # percent
        self.data = array.array('h') # signed short integer (-32768 to 32767) data
        self.sampleRate = 44100 # of samples per second (standard)
        self.numChan = 1 # of channels (1: mono, 2: stereo)
        self.dataSize = 2 # 2 bytes because of using signed short integers => bit depth = 16
        self.numSamplesPerCyc = int(self.sampleRate / self.freq)
        self.numSamples = self.sampleRate * self.duration

    def create_wave(self):
        for i in range(self.numSamples):
            v = 32767 * float(self.volume) / 100
            #sample *= math.sin(math.pi * 2 * (i % self.numSamplesPerCyc) / self.numSamplesPerCyc)
            sample = math.sin(math.pi * 2 * (i % self.numSamplesPerCyc) / self.numSamplesPerCyc)
            sample *= v
            self.data.append(int(sample))
        f = wave.open('SineWave_' + str(self.freq) + 'Hz.wav', 'w')
        f.setparams((self.numChan, self.dataSize, self.sampleRate, self.numSamples, "NONE", "Uncompressed"))
        f.writeframes(self.data.tostring())
        f.close()

    def plot(self):
        samplerate1, x = wavfile.read('SineWave_440Hz.wav')
        plt.figure(1)
        plt.title('Sine Wave...')

        plt.plot(x[:1000], linewidth=2, color='r')
        plt.xlabel("no of samples")
        plt.ylabel("amplitudes")
        plt.show()

