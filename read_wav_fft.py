import matplotlib.pyplot as plt

from scipy.io import wavfile

samplerate, data = wavfile.read("SineWave_440Hz.wav")
#samplerate, data = wavfile.read("slow.wav")
#samplerate, data = wavfile.read("fast.wav")

#plt.plot(data[:200])

#plt.show()

samples = data.shape[0]


from scipy.fftpack import fft,fftfreq

datafft = fft(data)
#Get the absolute value of real and complex component:
fftabs = abs(datafft)

freqs = fftfreq(samples,1/samplerate)

#plt.plot(freqs,fftabs)

#plt.show()

plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
print(max(freqs))
print(max(fftabs[:int(freqs.size/2)]))
plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])

plt.show()
