#!/usr/bin/env python
# coding: utf-8

# Import Necessary libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools


# In[3]:


df = pd.read_csv('news.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[7]:


df.duplicated().sum()


# In[8]:


# DAta flair - get the labels
labels = df.label
labels.head()


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(df['text'],labels,test_size=0.2,random_state=0)


# In[10]:


x_train,y_train


# In[11]:


x_test,y_test


# In[12]:


#initialize a tfidf_vectorizer
tfidfvectorizer = TfidfVectorizer(stop_words='english',max_df=.7)

#fit and transform  train set and transform test set

tfidf_train = tfidfvectorizer.fit_transform(x_train)
tfidf_test = tfidfvectorizer.transform(x_test)


# In[16]:


pac = PassiveAggressiveClassifier(max_iter =50)
pac.fit(tfidf_train,y_train)

y_pred = pac.predict(tfidf_test)
score =accuracy_score(y_test,y_pred)
print(f'Accuracy:{round(score*100,2)}%')


# In[17]:


#build confusion matrix

confusion_matrix(y_test,y_pred,labels=['FAKE','REAL'])


# so with this model, we have 568 true positives 613 true negatives 39 false positives and 47 false negatives.

# In[ ]:




