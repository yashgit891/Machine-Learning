# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 04:28:54 2020

@author: KIIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x= LabelEncoder()
X[:, 0]=labelencoder_x.fit_transform(X[:, 0])
onehotencoder=OneHotEncoder(categorical_features = [0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)