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
import pickle
import count_moods
from sklearn.svm import SVC

dataframe = pd.read_csv('dataset.csv')
dataset = dataframe.values

#taking out independent and dependent variable
X = dataset[:,2:]
Y = dataset[:,1:2]

#counting similar moods
print('Original')
count_moods.count_moods_of_songs(Y)

#feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#spliting dataset
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state = 41)


print('Training')
count_moods.count_moods_of_songs(y_train)

print('Testing')
count_moods.count_moods_of_songs(y_test)

new_clf = SVC(kernel = 'rbf')
new_clf = new_clf.fit(x_train, y_train)
y_pred = new_clf.predict(x_test)

print('training accuracy {0}'.format(new_clf.score(x_train, y_train)))
print('testing accuracy {0}'.format(accuracy_score(y_test,y_pred)))

print(new_clf)

#pickling our classifier object#pickling our classifier
file_name = 'svm_classifier'
outfile = open(file_name,'wb')
pickle.dump(new_clf,outfile)
outfile.close()

#43 7 58 15

cm = confusion_matrix(y_test,y_pred)
print(cm)

#plotting bar-graph of our model accuracy
result = []
a_h_r_s = [49,17,66,24]
for i in range(4):
    result.append((cm[i][i]/a_h_r_s[i])*100)

import results

results.plot_results(result,
                     'SVM',classes = ('angry', 'happy', 'relax', 'sad'))