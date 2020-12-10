# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:12:05 2019

@author: Prashant
"""
import pandas as pd
data = pd.read_csv("dataset.csv")
X = data.iloc[:,2:]  #independent columns
y = data.iloc[:,1:2]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


print(data[['cent_mean','mfcc_std','chroma_stft_mean','perc_std','poly_std','contrast_mean','spec_bw_mean','poly_var']])