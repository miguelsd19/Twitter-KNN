# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:52:26 2020

@author: Admin
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

archivo= "cleantweets.csv"

df=pd.read_csv(archivo,names=['sentiment','polarity','subjectivity'])

X = df.iloc[:,1:3].values
y = df.iloc[:,0].values

X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=11)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

knn=KNeighborsClassifier()

print(knn.fit(X=X_train,y=y_train))

predicted=knn.predict(X=X_test)

expected=y_test

print("Predicciones")
print(predicted[:20])

print("Expected")
print(expected[:20])

wrong=[(p,e) for (p,e) in zip(predicted,expected) if p!=e ]

print(wrong)

print("accuracy: ")
print(f'{(len(expected) - len(wrong)) / len(expected):.2%}')

print(f'{knn.score(X_test, y_test):.2}')

#Matriz de confucion

confusion= confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

#names=[str(digit) for digit in y]
print("")
print(classification_report(expected, predicted, target_names={"neutral","postive", "negative"}))

confusion_df=pd.DataFrame(confusion, index=range(3), columns=range(3))

axes=sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')








