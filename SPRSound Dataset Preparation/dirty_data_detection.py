# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 07:58:40 2026

@author: kiara
"""

from pathlib import Path
import pandas as pd
import os
import json
from pandas.api.types import is_integer_dtype
from IPython.display import display
import librosa
      
def read_json_files(folder_path):
    folder_path = Path(folder_path)
    all_data = []   # using this to collect data from each JSON file

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):   # Only process JSON files
            file_path = folder_path / file_name
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
    df["start"] = pd.to_numeric(df["start"].astype("string").str.strip(),errors="coerce").astype("Int64")
    assert is_integer_dtype(df["start"].dtype)
    #converting end to int
    df["end"] = pd.to_numeric(df["end"].astype("string").str.strip(),errors="coerce").astype("Int64")
    assert is_integer_dtype(df["end"].dtype)
    #making sure the data type was changed
    print(f"For the start category {df["start"].dtype}")  
    print(f"For the end category {df["end"].dtype}")   

    return final_df

def check_duplicate(df):
    #Detect any complete duplicates with the same start and end values
    duplicates = df.duplicated(subset=["source_file","start","end", "type"], keep=False )
    duplicated_rides = df.loc[duplicates].dropna(subset=["source_file", "start","end","type"]).sort_values("source_file")
    print("Complete Duplicates:")
    display(duplicated_rides[["source_file", "start", "end", "type"]])
    
    #Detect any partial duplicates with the same start and end values
    duplicates = df.duplicated(subset=["source_file","start","end"], keep=False )
    duplicated_rides = df.loc[duplicates].dropna(subset=["source_file", "start","end","type"]).sort_values("source_file")
    print("Start-End Duplicates:")
    display(duplicated_rides[["source_file", "start", "end", "type"]])
    
    #Detect any partial duplicates with the same start and type values
    duplicates = df.duplicated(subset=["source_file", "start","type"], keep=False )
    duplicated_rides = df.loc[duplicates].dropna(subset=["source_file", "start","end","type"]).sort_values("source_file")
    print("Start-Type Duplicates:")
    display(duplicated_rides[["source_file", "start", "end", "type"]])
    
    #Detect any partial duplicates with the same end and type values
    duplicates = df.duplicated(subset=["source_file", "end","type"], keep=False )
    duplicated_rides = df.loc[duplicates].dropna(subset=["source_file", "start","end","type"]).sort_values("source_file")
    print("End-Type Duplicates:")
    display(duplicated_rides[["source_file", "start", "end", "type"]])
      
def wav_files_len(wav_path):
    wav_path = Path(wav_path)
    all_data = []   # Collect data from each wav file

    # Loop through all files in the folder
    for file_name in os.listdir(wav_path):
        if file_name.endswith(".wav"):   # Only process wav files
            file_path = wav_path / file_name
            try:
                y, sr = librosa.load(file_path, sr=None)  # keep original sample rate
                length_ms = librosa.get_duration(y=y, sr=sr) * 1000  #finding duration in ms
                all_data.append({"source_file": file_name, "length_ms": length_ms})
            except FileNotFoundError:
                print(f"Error: The file at {file_path} was not found.")
            except PermissionError:
                print(f"Error: You don’t have permission to read {file_path}.")
            except Exception as e:
                print(f"Unexpected error: {e}")
    return pd.DataFrame(all_data)

def main():
    #saying the path to the training json folder in the SPRS folder
    file_folder = Path("C:/Users/kiara/Documents/ECNG/Y4S1/ECNG3020/SPRSound-main/BioCAS2022/train2022_json")    
    df = read_json_files(file_folder)
    
    #getting general information about the dataset
    print(df.info())
    print(df.head())
    print(df.describe())     
    
    #checking record categories for typing inconsistencies
    print(df["record_annotation"].unique())
    
    #checking event types for typing inconsistencies and making sure I have all the events
    t_count=df["type"].value_counts()
    print(t_count)         
    
    #checking the data type of the type, start and end categories
    print(f"For the type category {df["type"].unique()}")   
    print(f"For the start category {df["start"].unique()}")   
    print(f"For the end category {df["end"].unique()}")  
    
    #finding the average event duration for the standardization step in signal processing
    df["duration"] = df["end"]-df["start"]
    #print(df.info())
    print(df["duration"].describe())
    display(df[["start", "end", "duration"]].head(25))
    
    #looking at min and max of start and end category
    print(f"For the start category {df["start"].describe()}")   #checking the data type of the start category
    print(f"For the end category {df["end"].describe()}")   #checking the data type of the end category
    
    # #checking whether there are any out of range time data
    invalid_duration=((df["end"]-df["start"])<=0).sum()
    #dur_invalid=invalid_duration.sum()
    print(f"Number of invalid durations: {invalid_duration}")
    #validating that there are 6665 valid durations 
    invalid_duration=((df["end"]-df["start"])>0).sum()
    #dur_invalid=invalid_duration.sum()
    print(f"Number of valid durations: {invalid_duration}")
    
    #checking whether there are any partial or complete duplicates within one json file
    check_duplicate(df)
    
    #checking for any missing start or end values
    print("Data missing from start = ")
    missing_st = df.loc[df["start"].isna()]
    display(missing_st.describe(include="all"))
    
    print("Data missing from end = ")
    missing_e = df.loc[df["end"].isna()]
    display(missing_e.describe(include="all"))
    
    #ensuring that the start and end times are less than the audio duration
    #saying the path to the training json folder in the SPRS folder
    wav_path = Path("C:/Users/kiara/Documents/ECNG/Y4S1/ECNG3020/SPRSound-main/BioCAS2022/train2022_wav")    
    wav_df=wav_files_len(wav_path)
    
    # In df (JSON records), strip extension and keep only basename
    df["source_file"] = df["source_file"].str.replace(".json", ".wav")
    # In wav_df, ensure only basename
    wav_df["source_file"] = wav_df["source_file"].str.replace(".json", ".wav")
    
    #adding this to the orginal df
    merged_df = df.merge(
     wav_df,
     on="source_file",
     how="left",
     validate="many_to_one"
    )
    print(merged_df.info())
    display(merged_df[["source_file", "start", "end", "type", "length_ms"]].head(10))
   
    #print(merged_df["length_ms"].unique())     #checking the number of files for each audio length
    l_count=merged_df["length_ms"].value_counts()
    print(l_count)
    
    #checking that the end times are less than the duration
    invalid_end=(merged_df["end"] > merged_df["length_ms"]).sum()
    print(f"Number of invalid end times: {invalid_end}")
    
    #checking that the start times are less than the duration
    invalid_start=(merged_df["start"] > merged_df["length_ms"]).sum()
    print(f"Number of invalid start times: {invalid_start}")

main()
