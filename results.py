# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 00:07:30 2019

@author: Prashant
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_results(results,title,classes): 
    y_pos = np.arange(len(classes))
    performance = results 
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, classes)
    plt.ylabel('Accuracy')
    plt.title(title)
     
    plt.show()