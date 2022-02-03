#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


titanic_data = pd.read_csv('titanic_train.csv')


# In[3]:


titanic_data.head()


# In[4]:


titanic_data.shape


# In[6]:


titanic_data.info()


# In[7]:


titanic_data.isnull()


# In[10]:


sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='inferno')


# From the above Heatmap we can observe that there is almost 20% of the values present in Age are null and the proportion of missing
# values are small enough for reasonable replacement but it looks like too much is missing for the cabin column.

# In[11]:


sns.countplot(data=titanic_data,x='Survived')


# In[16]:


sns.set_style('whitegrid')
sns.countplot(data=titanic_data,x='Survived',hue='Sex')


# In[17]:


sns.countplot(data=titanic_data,x='Survived',hue='Pclass')


# In[24]:


sns.histplot(data=titanic_data,x='Fare',color='red',bins=40)


# In[36]:


sns.distplot(titanic_data['Age'].dropna(), color='magenta',bins=40)


# # Data cleaning

# In[37]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data=titanic_data,color='darkgreen')


# In[52]:


def impute_age(cols):
    Age= cols[0]
    Pclass =cols[1]
    if pd.isnull(Age):
   
        if Pclass ==1:
            return 37

        elif Pclass ==2 :
                return 29

        else:
            return 24
    else :
            return Age


# In[55]:


titanic_data['Age']=titanic_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[61]:


sns.heatmap(titanic_data.isnull(),cmap='inferno')


# In[62]:


titanic_data=titanic_data.drop('Cabin',axis=1)


# In[63]:


titanic_data.head()


# 
# # converting categorical variables

# In[65]:


sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
embarked=pd.get_dummies(titanic_data['Embarked'],drop_first=True)


# In[71]:


titanic= titanic_data.drop(['PassengerId','Name','Ticket','Sex','Embarked'],axis=1)


# In[72]:


new_titanic=pd.concat([titanic,sex,embarked],axis=1)


# In[73]:


new_titanic.head()


# In[75]:


x=new_titanic.drop(['Survived'],axis=1)
y=new_titanic['Survived']


# In[76]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=.30,random_state=100)


# In[77]:


x_train
y_train


# In[92]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[93]:


logmodel=lr.fit(x_train,y_train)


# In[94]:


pred =lr.predict(x_test)


# In[95]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics  import accuracy_score


# In[96]:


accuracy=confusion_matrix(y_test,pred)
accuracy


# In[97]:


accuracy= accuracy_score(y_test,pred)


# In[98]:


accuracy


# In[ ]:




