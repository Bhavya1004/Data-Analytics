#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[106]:


df = pd.read_csv("heart.csv")


# In[107]:


df.head()


# In[108]:


df.shape


# In[109]:


df.info()


# In[110]:


df.head(-5)


# In[111]:


df.isnull().sum()


# In[112]:


df.duplicated().sum()


# In[113]:


df.describe()


# In[114]:


sns.pairplot(df)


# In[115]:


corr = df.corr(method='pearson')
plt.figure(figsize=(10,6))
heatmap=sns.heatmap(corr,annot=True,cmap='inferno')
heatmap.set_title('Correlation Between variables')
plt.show()


# There is no such correlation found between any of the variables

# In[116]:


sns.countplot(data=df,x='HeartDisease',hue='Sex')
# From Here we can observe that above 50% of the data belongs to male and male people have more heart diseases than the female.


# In[117]:


sns.distplot(df['Age'],color='magenta',bins=25)


# In[118]:


people_with_hd = df[df['HeartDisease']==1]['Age'].count()
Total = df['HeartDisease'].count()
percentage = print('{} People With HeartDisease {}%'.format(people_with_hd,round(people_with_hd/Total*100)))


# In[119]:


df.head()


# In[120]:


df_numerical =df.drop(['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'],axis=1)


# In[121]:


plt.figure(figsize=(10,6))
box =sns.boxplot(data=df_numerical)
box.set_title('Boxplot For All Numerical Variables')


# From the above BoxPlot we can observe that there is no such outliers found for Age,FastingBS,MaxHR,oldpeak but the outliers for
# RestingBp and chlesterol are more.Outliers treatment should be done these two columns so that we can get good accuracy score.

# In[123]:


for feature in df_numerical:
    data= df.copy()
    data[feature].hist(bins=20)
    plt.xlabel =(feature)
    plt.ylabel =('count')
    plt.title(feature)
    plt.show()


# In[97]:


df.head()


# In[188]:


dummy =pd.get_dummies(df,columns=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'])
#coverting the categorical column into One hot coding so that we can apply the Ml algorithm.


# In[189]:


dummy


#  As we have observed outliers in cholestrol and RestingBp we have to treat them before applying any model

# In[126]:


df['Cholesterol'].describe()


# # Removing Outliers

# In[127]:


Q1 = 173.25
Q3 = 267
IQR = Q3 - Q1
print(Q3 -1.5*IQR)
print(Q1 + 1.5*IQR)


# In[167]:


new_df = dummy[(dummy['Cholesterol'] > 313.875) | (dummy['Cholesterol'] < 126.375)]
print(new_df)


# In[168]:


sns.boxplot(new_df['Cholesterol'])


# In[132]:


df['RestingBP'].describeibe()


# In[133]:


plt.hist(df['RestingBP'])


# In[144]:


Q1=120
Q3=140
IQR =Q3-Q1
print(Q3-1.5*IQR)
print(Q1+1.5*IQR)


# In[169]:


new_df2=dummy[(dummy['RestingBP']>150)|(dummy['RestingBP']<110)]


# In[170]:


plt.boxplot(new_df2['RestingBP'])


# In[190]:


#splitting the data into train and test


# In[171]:


x = new_df2.drop(['HeartDisease'],axis=1)
y =new_df2['HeartDisease']


# In[172]:


x,y


# In[173]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2)


# In[174]:


x_train


# # Implementing Models to predict Heartdisease

# In[175]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[178]:


lr.fit(x_train,y_train)


# In[179]:


pred= lr.predict(x_test)


# In[184]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# In[187]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
accuracy_score(y_test,y_pred)


# In[ ]:




