# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:36:19 2026

@author: kiara

Purpose: This code takes the full SPRSound 2022 training json and wav folder.
It does the following functions: 
    - Reads in the json file
    - Filters out the poor quality records
    - Verfies that each json file has an associated wav file
    - Uses the start and end timestamps to segment the audio
    - Filters each segment using a 5th-order Butterworth bandpass filter
    - Normalizes the amplitude of each segment using z-score normalization
    - Standardizes the segment length to the average segment length.
        ~ If it is shorter then zero padding is used.
        ~ If it is longer then a sliding window is used.
    - Provides a count of every missing wav file and poor quality record that was skipped.
"""

from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pandas.api.types import is_integer_dtype
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt
from scipy.stats import zscore

def read_json_file(file_path):
    all_data = []   # using this to collect data from each JSON file

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # If JSON is a list of dicts, extend the list
            if isinstance(data, list):
                all_data.extend(data)
            # If JSON is a single dict, append it
            elif isinstance(data, dict):
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

        segments.append({
            "signal": segment,
            "type": row["type"],
            "start_ms": row["start"],
            "end_ms": row["end"] })
    
    return segments

def std_data(segments, sr, avg):
    #moving through each segment in the segment dict
    for seg in segments:
        #calculating event duration
        dur=seg["end_ms"] - seg["start_ms"]
        #print(f"dur= {dur}")        #checking duration before it assesses it
        if (dur<avg):
            #print("segment is less than 1.3")
            pad_dur=avg-dur        #finding out how much time I need to add in to hit 1.3s
            pad_no= int(pad_dur * sr / 1000)    #calculating how much frames I need to get the 1.3s
            #zero padding
            seg["signal"]=np.pad(
                seg["signal"], 
                pad_width=(0,pad_no),
                mode="constant")
            #show_wav(seg["signal"])
        elif (dur>avg):
            #print(f"segment is more than 1.3: {dur}")
            window_size = int(avg * sr / 1000)
            seg["signal"]= sliding_window_view(seg["signal"], window_shape=window_size)
            
    return segments

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
    wav_folder = Path("C:/Users/kiara/Documents/ECNG/Y4S1/ECNG3020/SPRSound-main/BioCAS2022/train2022_wav")    
    json_folder = Path("C:/Users/kiara/Documents/ECNG/Y4S1/ECNG3020/SPRSound-main/BioCAS2022/train2022_json")    
    
    pq_count=0      #counting rejected poor quality records
    missing_count=0 #counting any json files missing a wav file
    
    # Loop through all files in the json folder
    for json_path in json_folder.glob("*.json"):  # Only process JSON files
        try:
            #skipping poor quality files
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("record_annotation") == "Poor Quality":
                #print(f"Skipping {json_path.name}: Poor Quality")
                pq_count=pq_count+1
                continue
            
            #since the wav files and json files share the same name
            wav_path = wav_folder / f"{json_path.stem}.wav"
            #ensuring that the json file has an associated wav file
            if not wav_path.exists():
                missing_count=missing_count+1
                #print(f"No matching wav for {json_path.name}")
                continue
            
            #extracting json and wav info
            #print(f"Processing {json_path.name}")
            df = read_json_file(json_path)
            data, sr = librosa.load(wav_path, sr=8000)  #sampling the audio file at 8kHz
            
            #signal processing
            segments= segment_data(data, df, sr)
            for seg in segments:
                seg["signal"]=wav_filter(seg["signal"],sr)        #send the sampled data to the butterworth filter
                seg["signal"]=zscore(seg["signal"])               #normalizing the amplitude of the filtered segment
                #show_compare(seg["signal"],std_data,json_path.name,seg["type"])
            SPRS_avg=1300
            segments=std_data(segments,sr, SPRS_avg)
            #print(segments)
    
        except FileNotFoundError:
            print(f"Error: The file at {wav_path} was not found.")
        except PermissionError:
            print(f"Error: You don’t have permission to read {wav_path}.")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
    print(f"Number of rejected poor-quality records: {pq_count}")
    print(f"Number of missing wav files: {missing_count}")
        
main()
