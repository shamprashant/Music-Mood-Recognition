# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:22:13 2019

@author: Prashant
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from mpl_toolkits.mplot3d import Axes3D

def visualiaze_dataset(x_train,y_train):
    pca = KernelPCA(n_components=3,kernel='rbf')
    x_train = pca.fit_transform(x_train)
    print(x_train.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train[y_train[:,0] == 'sad',0],x_train[y_train[:,0] == 'sad',1],
                x_train[y_train[:,0] == 'sad',2],label = 'sad',c='blue',marker = '*')
    
    ax.scatter(x_train[y_train[:,0] == 'happy',0],x_train[y_train[:,0] == 'happy',1],
                x_train[y_train[:,0] == 'happy',2],label = 'happy',c='green',marker = '+')
    
    ax.scatter(x_train[y_train[:,0] == 'relax',0],x_train[y_train[:,0] == 'relax',1],
                x_train[y_train[:,0] == 'relax',2],label = 'relax',c='orange',marker = '3')
    
    ax.scatter(x_train[y_train[:,0] == 'angry',0],x_train[y_train[:,0] == 'angry',1],
                x_train[y_train[:,0] == 'angry',2],label = 'angry',c='red',marker='x')
    plt.legend()
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_zlabel('component 3')
    ax.set_title('Scattter plot of Training DataSet')
    
    plt.show()