# author: Ovidiu Mura
# date: April 30, 2019

# Reference: https://www.johndcook.com/blog/2016/03/10/creating-police-siren-sounds-with-frequency-modulation/
import sys
from scipy.io.wavfile import write
from numpy import arange, pi, sin, int32, int16
import configparser
import numpy as np
import matplotlib.pyplot as plt
import wave
from read_wav_fft import READ_WAV_FFT
from mix_sounds import *
from sine import *
from filters import *

# from scikits import wavread
from noise import *
import argparse

# This file contains the implementation of the eSound - the emergency sound which generates frequency modulation signal
# and it stores in a wav file, and the method to plot the signal visually.
class eSound:
    def __init__(self):
        self.N = 48000
        self.file_name = 'eSound.wav'

    # frequency modulation method, which generates signal with modulated frequency.
    # returns the modulated frequency
    # t - time
    # cf - carrier frequency
    # mf - modulation frequency
    # beta - modulation index
    def freq_modulation(self, t, cf, mf, beta):
        return sin(2*pi*cf*t - beta*sin(2*mf*pi*t))

    # It converts the signal to int fo 2 bytes and amplifies the signals.
    def to_int16(self, signal):
        # Take samples in [-1, 1] and scale to 16-bit integers,
        # values between -2^15 and 2^15 - 1.
        return int16(signal*(2**15-1))

    # It plots the signal of emergency sound read from a file.
    # file_name - the name of file to read the signal
    def plot_wave(self, file_name='eSound.wav'):
        spf = wave.open(file_name,'r')

        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, 'int32')
        plt.figure(1)
        plt.title('Emergency Sound Wave...')
        plt.plot(signal[0:1600])
        plt.xlabel("no of samples")
        plt.ylabel("amplitudes")
        plt.show()

    # It creates the emergency sound.
    def create_emergency_sound(self):
        x = arange(0,3*self.N,0.5,float) # three seconds of audio
        data = self.freq_modulation(x/self.N, 1000, 8, 100) # modulated frequency data
        write(self.file_name, self.N, self.to_int16(data)) # write the data after amplify the signal in int16

def main():
    # initialize the config parser
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    project_name = config.get('DEFAULT','PROJECT_NAME')

    # create the command line arguments for the implemented functionality
    parser = argparse.ArgumentParser(description="Emergency Sound Detection")
    parser.add_argument('--create', '-create', dest='wav', default='none', help='Use [<all>] to create wave files: emergency sound, white noise, pink noise, and brown noise')
    parser.add_argument('--avg_mix_plot', '-avg_mix_plot', dest='avg_mix_plot', default='none', help='Use any [<brown>, <white>, <pink>] mix emergency signal with the brown, white, or pink noise'
                                                                                     '\nUse any [<plot>, <plot_white>, <plot_brown>, <plot_pink> ] to mix the emergency sound with the corresponding noise and plot')
    parser.add_argument('--is_emergency_signal_in_mix', '-is_emergency_signal_in_mix', dest='noise_type', default='none', help='Use [<brown>, <white>, <pink>] to search the emergency signal in the mixed signal with the corresponding noise')
    parser.add_argument('--convolve', '-convolve', dest='convolve', default='none', help='Use [<esound>] to convolve emergency sound with the Gaussian window, and Convolution Theorem')
    parser.add_argument('--freq_domain', '-freq_domain', dest='freq_domain', default='none', help='Use [<esound>, <brown>, <white>, <pink>] to project the signals in the frequency domain\n')
    parser.add_argument('--filter', '-filter', dest='filter', default='none', help='Use [<filters>] to run the low, high, band, and goertzel filters on a sine wave')
    args = parser.parse_args()

    if (len(sys.argv)==1):
        parser.print_usage()
        exit(1)

    # read the files of mixed signals from config file
    output_mix_1 = config.get('MIXED_SIGNALS', 'eSOUND_bNOISE')[1:-1]
    output_mix_2 = config.get('MIXED_SIGNALS', 'eSOUND_wNOISE')[1:-1]
    output_mix_3 = config.get('MIXED_SIGNALS', 'eSOUND_pNOISE')[1:-1]

    mix = Mix()

    print("Welcome to " + str(project_name) + "!")

    # generate the emergency and the white, pink, and brown noise
    if(args.wav is not 'none' and args.wav in ('all')):
        print("info: creating emergency sound, white noise, pink noise, brown noise!")

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

        s = Sine()
        s.create_wave()
        s.plot()
        print('info: wav files (eSound.wav, bNoise.wav, pNoise.wav, wNoise.wav), successfully created')

    # mix and plot the emergency signal with the noise signal
    elif(args.avg_mix_plot in ('plot', 'brown', 'white', 'pink', 'plot_white', 'plot_brown', 'plot_pink')):
        if(args.avg_mix_plot == 'brown'):
            mix.avg_mix('eSound.wav', 'bNoise.wav', output_mix_1)
        elif (args.avg_mix_plot == 'white'):
            mix.avg_mix('eSound.wav', 'wNoise.wav', output_mix_2)
        elif (args.avg_mix_plot == 'pink'):
            mix.avg_mix('eSound.wav', 'pNoise.wav', output_mix_3)
        elif (args.avg_mix_plot == 'plot_brown'):
            mix.avg_mix('eSound.wav', 'bNoise.wav', output_mix_1)
            mix.plot_avg_mix()
        elif (args.avg_mix_plot == "plot_white"):
            mix.avg_mix('eSound.wav', 'wNoise.wav', output_mix_2)
            mix.plot_avg_mix()
        elif (args.avg_mix_plot == 'plot_pink'):
            mix.avg_mix('eSound.wav', 'pNoise.wav', output_mix_3)
            mix.plot_avg_mix()
    # check if the signals have relation and plot the results
    elif (args.noise_type in ('white', 'pink', 'brown', 'plot_white', 'plot_pink', 'plot_brown')):
        if(args.noise_type == 'brown'):
            if(mix.is_in_mix('eSound.wav', 'bNoise.wav', output_mix_1) == True):
                print("info: Emergency Sound found in the mix signal!")
            # mix.plot_mix_and_original_signal()
        elif(args.noise_type == 'white'):
            if(mix.is_in_mix('eSound.wav', 'wNoise.wav', output_mix_2) == True):
                print("info: Emergency Sound found in the mix signal!")
                # mix.plot_mix_and_original_signal()
        elif(args.noise_type == 'pink'):
            if(mix.is_in_mix('eSound.wav', 'pNoise.wav', output_mix_3) == True):
                print("info: Emergency Sound found in the mix signal!")
                # mix.plot_mix_and_original_signal()
        elif (args.noise_type == 'plot_brown'):
            if(mix.is_in_mix('eSound.wav', 'bNoise.wav', output_mix_1) == True):
                print("info: Emergency Sound found in the mix signal!")
                mix.plot_avg_mix()
        elif (args.noise_type == 'plot_white'):
            if(mix.is_in_mix('eSound.wav', 'wNoise.wav', output_mix_2) == True):
                print("info: Emergency Sound found in the mix signal!")
                mix.plot_avg_mix()
        elif (args.noise_type == 'plot_pink'):
            if(mix.is_in_mix('eSound.wav', 'pNoise.wav', output_mix_3) == True):
                print("info: Emergency Sound found in the mix signal!")
                mix.plot_avg_mix()
    # convolve the emergency and the noise signals
    elif (args.convolve in ('esound', 'brown', 'white', 'pink')):
        conv = Convolute()
        if(args.convolve == 'esound'):
            conv.convolve_gaussian_window()
            conv.fft_convolve()
        elif(args.convolve == 'brown'):
            conv.fft_convolve('eSound.wav', 'bNoise.wav')
        elif(args.convolve == 'white'):
            conv.fft_convolve('eSound.wav', 'wNoise.wav')
        elif(args.convolve == 'pink'):
            conv.fft_convolve('eSound.wav', 'pNoise.wav')
    # plot the emergency, and noise signals in frequency domain
    elif (args.freq_domain in ('esound', 'brown', 'white', 'pink')):
        rwf = READ_WAV_FFT()
        if(args.freq_domain == 'esound'):
            rwf.read_wav_fft('eSound.wav')
            rwf.plot()
        elif(args.freq_domain == 'brown'):
            rwf.read_wav_fft('bNoise.wav')
            rwf.plot()
        elif(args.freq_domain == 'white'):
            rwf.read_wav_fft('wNoise.wav')
            rwf.plot()
        elif(args.freq_domain == 'pink'):
            rwf.read_wav_fft('pNoise.wav')
            rwf.plot()
    # run the filters on a sine wave
    elif(args.filter is not 'none' and args.filter in ('filters')):
        get_sine_freq()
        f = Filter()
        f.lowpass()
        f.highpass()
        f.bandpass()

# the starting point of execution of the program
if __name__ == "__main__":
    main()
