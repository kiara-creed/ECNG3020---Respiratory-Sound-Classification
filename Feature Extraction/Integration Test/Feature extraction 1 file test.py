# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:21:39 2026

@author: kiara

This integrates the signal processing code with the MFCC feauture extraction code. 
        
Purpose: This code takes 1 json and its respective wav file. It has the following pipeline:
         read in json file => resample wav file => segmentation => filter => normalize => standardization => mfcc feature extraction
        
"""

from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pandas.api.types import is_integer_dtype
from scipy.signal import butter, filtfilt
from scipy.stats import zscore

def read_json_files(file_path):
    file_name = file_path.name
    all_data = []   # using this to collect data from each JSON file

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # If JSON is a list of dicts, extend the list
            if isinstance(data, list):
                for d in data:
                    d["source_file"] = file_name
                all_data.extend(data)
            # If JSON is a single dict, append it
            elif isinstance(data, dict):
                data["source_file"] = file_name
                all_data.append(data)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except PermissionError:
        print(f"Error: You don’t have permission to read {file_path}.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    # Convert the collected list of dicts into a pandas DataFrame
    df = pd.DataFrame(all_data)
    # Expand the list of dicts in 'event_annotation' into separate rows
    df_exploded = df.explode("event_annotation").reset_index(drop=True)
    # Normalize the nested dicts into columns
    event_df = pd.json_normalize(df_exploded["event_annotation"])
    # Merge back with the original record_annotation if needed
    final_df = pd.concat([df_exploded.drop(columns=["event_annotation"]), event_df], axis=1)

    #Converting start and end timestamps to int
    #converting start to int
    final_df["start"] = pd.to_numeric(final_df["start"].astype("string").str.strip(),errors="coerce").astype("Int64")
    assert is_integer_dtype(final_df["start"].dtype)
    #converting end to int
    final_df["end"] = pd.to_numeric(final_df["end"].astype("string").str.strip(),errors="coerce").astype("Int64")
    assert is_integer_dtype(final_df["end"].dtype) 

    return final_df


def segment_data(audio, df, sr):
    segments = []
    #iterating through the timestamps in the json file
    for _, row in df.iterrows():
        #finding the array index cooresponding to the start and end timestamps
        start_sample = int(row["start"] * sr / 1000)
        end_sample = int(row["end"] * sr / 1000)

        #sgmenting the audio
        segment = audio[start_sample:end_sample]

        segments.append({"signal": segment,
                        "type": row["type"],
                        "start_ms": row["start"],
                        "end_ms": row["end"] })
    return segments

def std_data(segment, sr, avg, start, end):
    #calculating event duration
    dur=end-start
    print(f"dur= {dur}")        #checking duration before it assesses it
    if (dur<avg):
        #print("segment is less than 1.3")
        pad_dur=avg-dur        #finding out how much time I need to add in to hit 1.3s
        pad_no= int(pad_dur * sr / 1000)    #calculating how much frames I need to get the 1.3s
        #zero padding
        segment=np.pad( segment, pad_width=(0,pad_no), mode="constant")
        #show_wav(seg["signal"])
    elif (dur>avg):
        #print(f"segment is more than 1.3: {dur}")
        #truncation/ cutting off the end of the segment
        window_size = int(avg * sr / 1000)
        segment = segment[:window_size]
            
    return segment

def wav_filter(data,fs):
    #source: https://www.geeksforgeeks.org/data-science/signal-filtering-with-scipy/
    lowcut=50         #specifying the lower cut off
    highcut=2500        #specifying the higher cut off
    order=5     #order of filter
    
    nyq = 0.5 * fs      #finding nyquist frequency
    low = lowcut / nyq  #lower end of the critical frequency
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')     #butterworth bandpass filter
    
    return filtfilt(b, a, data) 

def plot_mel(mfccs,hop_l,sr): 
    plt.figure(figsize=(10, 5))
    librosa.display.specshow( mfccs, 
                             sr=sr,
                             x_axis='time', 
                             cmap='inferno', 
                             hop_length=hop_l)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    plt.show()

def main():
    #saying the path to the training json folder in the SPRS folder
    wav_path = Path("SPRSound-main/BioCAS2022/train2022_wav/40638274_9.7_1_p3_1765.wav")    
    file_folder = Path("SPRSound-main/BioCAS2022/train2022_json/40638274_9.7_1_p3_1765.json")    
    
    df = read_json_files(file_folder)
    
    try:
        #loading the audio file
        data, sr = librosa.load(wav_path, sr=8000)  #sampling the audio file at 8kHz
        
        #chosen standard duration each event for the SPRSound dataset
        dur=2000
        #converting the 25ms window to a number of samples
        frame_length_ms = 25
        win_length = int(frame_length_ms * sr / 1000)
        #converting the 5ms window to a number of samples
        hop_length_ms = 5
        hop_length = int(hop_length_ms * sr / 1000)
        
        #segmenting the audio using the timestamps
        segments= segment_data(data, df, sr)
        
        for seg in segments:
            #filtering signal
            seg["signal"]=wav_filter(seg["signal"],sr)
            #normalizing signal
            seg["signal"]=zscore(seg["signal"])
            #standardizing the segments using the average event duration
            seg["signal"]=std_data(seg["signal"],sr, dur, seg["start_ms"], seg["end_ms"])
            #performing the mfcc feature extraction
            mfcc = librosa.feature.mfcc(y=seg["signal"],
                                        sr=sr,
                                        n_mfcc=30,
                                        n_fft=256,
                                        win_length=win_length,
                                        hop_length=hop_length,
                                        n_mels=20 )
            plot_mel(mfcc,hop_length,sr);
    except FileNotFoundError:
        print(f"Error: The file at {wav_path} was not found.")
    except PermissionError:
        print(f"Error: You don’t have permission to read {wav_path}.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
main()

