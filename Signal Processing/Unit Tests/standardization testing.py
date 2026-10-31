# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:46:33 2026

@author: kiara
"""

from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pandas.api.types import is_integer_dtype
from numpy.lib.stride_tricks import sliding_window_view

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
    final_df["start"] = pd.to_numeric(
     final_df["start"]
     .astype("string")
     .str.strip(),
     errors="coerce"
    ).astype("Int64")
    assert is_integer_dtype(final_df["start"].dtype)
    #converting end to int
    final_df["end"] = pd.to_numeric(
     final_df["end"]
     .astype("string")
     .str.strip(),
     errors="coerce"
    ).astype("Int64")
    assert is_integer_dtype(final_df["end"].dtype)
    #making sure the data type was changed
    print(f"For the start category {final_df["start"].dtype}")  
    print(f"For the end category {final_df["end"].dtype}")   

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
        print(f"dur= {dur}")        #checking duration before it assesses it
        if (dur<avg):
            print("segment is less than 1.3")
            pad_dur=avg-dur        #finding out how much time I need to add in to hit 1.3s
            pad_no= int(pad_dur * sr / 1000)    #calculating how much frames I need to get the 1.3s
            #zero padding
            seg["signal"]=np.pad(
                seg["signal"], 
                pad_width=(0,pad_no),
                mode="constant")
            show_wav(seg["signal"])
        elif (dur>avg):
            print(f"segment is more than 1.3: {dur}")
            seg["signal"]= sliding_window_view(seg["signal"], window_shape=avg)
            
    return segments

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
    

def main():
    #saying the path to the training json folder in the SPRS folder
    wav_path = Path("C:/Users/kiara/Documents/ECNG/Y4S1/ECNG3020/SPRSound-main/BioCAS2022/train2022_wav/40638274_9.7_1_p3_1765.wav")    
    file_folder = Path("C:/Users/kiara/Documents/ECNG/Y4S1/ECNG3020/SPRSound-main/BioCAS2022/train2022_json/40638274_9.7_1_p3_1765.json")    
    
    df = read_json_files(file_folder)
    
    try:
        data, sr = librosa.load(wav_path, sr=8000)  #sampling the audio file at 8kHz
        print("sr: ",sr)     #print the sr to make sure it was right
        #print("number of samples: ", data.shape)
        segments= segment_data(data, df, sr)
        print(segments)
        for seg in segments:
            show_wav(seg["signal"])
        SPRS_avg=1300
        std_data(segments,sr, SPRS_avg)
        print(segments)
    except FileNotFoundError:
        print(f"Error: The file at {wav_path} was not found.")
    except PermissionError:
        print(f"Error: You don’t have permission to read {wav_path}.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
main()
