# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:08:08 2026

@author: kiara
"""

from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import signal
      
def wav_filter(data):
    #source: https://www.geeksforgeeks.org/data-science/signal-filtering-with-scipy/
    fs=8000         #sample frequency
    lowcut=50         #specifying the lower cut off
    highcut=2500        #specifying the higher cut off
    order=5     #order of filter
    
    nyq = 0.5 * fs      #finding nyquist frequency
    low = lowcut / nyq  #lower end of the critical frequency
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')     #butterworth bandpass filter
    
    #plotting the frequency response of the filter
    #modified code from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    w, h = signal.freqz(b, a, worN=8000)
    frequencies = w * fs / (2 * np.pi)      # Convert from radians/sample to Hz
    plt.figure(figsize=(10,5))

    #for a steep start and curved drop off
    plt.plot(frequencies, 20 * np.log10(np.maximum(np.abs(h), 1e-10)))
    #for a more gradual start and steep drop off
    #plt.semilogx(frequencies, 20*np.log10(np.maximum(np.abs(h),1e-10)))

    plt.title('5th Order Butterworth Bandpass Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(which='both', axis='both')
    plt.axvline(lowcut,
                color='red',
                linestyle='--',
                label=f'Low Cutoff ({lowcut} Hz)')
    plt.axvline(highcut,
                color='green',
                linestyle='--',
                label=f'High Cutoff ({highcut} Hz)')

    plt.legend()
    plt.show()
    
    return filtfilt(b, a, data)

def show_wav(data):
    #source: https://www.geeksforgeeks.org/data-science/signal-filtering-with-scipy/
    t = np.arange(len(data)) / 8000
    #plotting fitered and unfiltered signal
    plt.figure(figsize=(10,4))
    plt.plot(t, data, label='Original Signal', color='red')
    plt.title('Wave Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
def show_compare(data, filtered):
    #source: https://www.geeksforgeeks.org/data-science/signal-filtering-with-scipy/
    t = np.arange(len(data)) / 8000
    #plotting fitered and unfiltered signal
    plt.figure(figsize=(10,4))
    plt.plot(t, data, label='Original Signal', color='red', alpha=0.7)
    plt.plot(t, filtered, label='Band-pass Filtered', color='blue',  alpha=0.7)
    plt.title('5th Order Butterworth Bandpass Filter Signals')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def main():
    #saying the path to the training json folder in the SPRS folder
    wav_path = Path("C:/Users/kiara/Documents/ECNG/Y4S1/ECNG3020/SPRSound-main/BioCAS2022/train2022_wav/40919639_4.9_0_p4_1793.wav")    
    try:
        data, sr = librosa.load(wav_path, sr=8000)  #sampling the audio file at 8kHz
        print("sr: ",sr)     #print the sr to make sure it was right
        print("number of samples: ", data.shape)
        show_wav(data)
        filtered=wav_filter(data)        #send the sampled data to the butterworth filter
        show_compare(data,filtered)
        #plot my graphs
    except FileNotFoundError:
        print(f"Error: The file at {wav_path} was not found.")
    except PermissionError:
        print(f"Error: You don’t have permission to read {wav_path}.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    

main()
