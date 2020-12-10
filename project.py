# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:15:02 2019

@author: Prashant
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle

dataframe = pd.read_csv('dataset.csv')
dataset = dataframe.values

#taking out independent and dependent variables
X = dataset[:,2:]
Y = dataset[:,1:2]

#feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#spliting dataset
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state = 26)

#training model for k = 1,2,3.....25
train_acc_list = []
test_acc_list = []
for neighbour in range(1,26):
    
    clf = KNeighborsClassifier(n_neighbors=neighbour)
    clf = clf.fit(x_train, y_train)
    
    #storing predictions of our model
    y_pred = clf.predict(x_test)
    
    #training accuracy for n neighbour
    train_acc = clf.score(x_train,y_train)
    train_acc_list.append(train_acc)
    
    #testing accuracy for n neighbour
    test_acc = accuracy_score(y_test,y_pred)
    test_acc_list.append(test_acc)
    
#converting it into numpy array
train_acc_list = np.array(train_acc_list) * 100
test_acc_list = np.array(test_acc_list) *100

#plotting training accuracy and testing accuracy
neighbours_list = [i for i in range(1,26)]
plt.plot(train_acc_list, label = 'Training accuracy')
plt.plot(test_acc_list, label = 'Testing accuracy')
plt.xticks()
plt.xlabel('Neighbours')
plt.ylabel('Accuracy %')
plt.legend()
plt.show()

for n in neighbours_list:
    print('Accuracy for n_neighbour = {0} '.format(n))
    print('Training accuracy {0}'.format(train_acc_list[n-1]))
    print('Testing accuracy {0}'.format(test_acc_list[n-1]))
    print('\n')
    
#17 is best.
print(clf)
#clf = KNeighborsClassifier(n_neighbors=17)
#clf.fit(x_train,y_train)

#pickling our classifier
import os#must import this library
if os.path.exists('knn_classifier'):
        os.remove('knn_classifier') #this deletes the file
else:
    file_name = 'knn_classifier'
    outfile = open(file_name,'wb')
    pickle.dump(clf,outfile)
    outfile.close()

#unpickling our classifier
#infile = open(file_name,'rb')
#new_clf = pickle.load(infile)
#infile.close()

cm = confusion_matrix(y_test,y_pred)
print(cm)

#plotting bar-graph of our model accuracy
result = []
a_h_r_s = [49,17,66,24]
for i in range(4):
    result.append((cm[i][i]/a_h_r_s[i])*100)

import results

#results.plot_results(result,
 #                    'KNN Classifier',classes = ('angry', 'happy', 'relax', 'sad'))
#40 9 60 12