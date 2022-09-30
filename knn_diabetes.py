#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('/home/arjun/Downloads/diabetes.csv')
df.head(10)


# In[3]:


#setting input and output label
x=df.iloc[:,:-1].values 
y=df.iloc[:,-1].values


# #set x as input label
# x=df.iloc[:-1].values
# print(x)

# In[5]:


#to split data into training data and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
print(x_train)


# In[6]:


#Normalization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# In[8]:


#Applying Model
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)


# In[9]:


y_pred=classifier.predict(x_test)


# In[10]:


print(classifier.predict([[5,148,72,35,94,28,0.167,25]]))


# In[12]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print('confiusion matrix',confusion_matrix(y_test,y_pred))
print('accuracy score',accuracy_score(y_test,y_pred))
print('classification report',classification_report(y_test,y_pred))

