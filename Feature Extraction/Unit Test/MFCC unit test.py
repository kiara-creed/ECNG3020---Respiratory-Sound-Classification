# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:03:19 2026

@author: kiara

Purpose: This code takes 1 SRPSound .wav file, extracts the features using MFCC and plots a spectrogram
"""

from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_mel(mfccs): 
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mfccs, x_axis='time', cmap='inferno', hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    plt.show()
      
def main():
    #saying the path to the training json folder in the SPRS folder
    wav_path = Path("C:/Users/kiara/Documents/ECNG/Y4S1/ECNG3020/SPRSound-main/BioCAS2022/train2022_wav/40919639_4.9_0_p4_1793.wav")    
    
    try:
        data, sr = librosa.load(wav_path, sr=8000)  #sampling the audio file at 8kHz
        #converting the 25ms window to a number of samples
        frame_length_ms = 25
        win_length = int(frame_length_ms * sr / 1000)
        
        #converting the 5ms window to a number of samples
        hop_length_ms = 5
        hop_length = int(hop_length_ms * sr / 1000)
        
        #performing the mfcc feature extraction
        mfcc = librosa.feature.mfcc(
            y=data,
            sr=sr,
            n_mfcc=30,
            n_fft=256,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=20 )
        plot_mel(mfcc);
        
    except FileNotFoundError:
        print(f"Error: The file at {wav_path} was not found.")
    except PermissionError:
        print(f"Error: You don’t have permission to read {wav_path}.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    

main()
