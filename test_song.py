
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:00:06 2019

@author: Prashant
"""

import feature_extraction
import warnings
import pickle
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

def test_mood_of_song(path):
    print(path)
    
    #getting song name
    song_name = ''
    for char in path[::-1]:
        if char == '\\':
            break
        else:
            song_name = char + song_name
            
    messagebox.showinfo("Feature Extraction", '\n Extracting features of selected song : {0}'.format(song_name))
    
    print('\n Extracting features of selected song : {0}'.format(song_name))
    print('....')
    print('....')
    print('....')
    print('....')
    feature_extraction.extract_feature(path)
    
    #unpickling our object
    infile = open('logistic_classifier','rb')
    new_clf = pickle.load(infile)
    infile.close()
    
    #finding mood of our song
    original_dataset = pd.read_csv('dataset.csv').values[:,2:]
    original_dataset = original_dataset
    new_song_data = pd.read_csv('test_song_mood.csv').values[:,2:]
    X = np.concatenate((original_dataset,new_song_data))
    #scaling our features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    mood = new_clf.predict(X[-1:])
    print('\nMood of Song is : {0}'.format(mood))
    return mood