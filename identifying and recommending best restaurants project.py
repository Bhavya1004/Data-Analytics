#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas asd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


pip install WordCloud


# In[3]:


pip install matplotlib_venn


# In[4]:


df = pd.read_excel('data.xlsx')


# In[5]:


df.head()


# In[6]:


df1 = pd.read_excel('Country-Code.xlsx')


# In[7]:


df1.head()


# In[8]:


df1.columns = df1.columns.str.replace(' ','_')


# In[9]:


df.isnull().sum()


# In[10]:


df['Cuisines'] = df['Cuisines'].fillna('Others')


# In[11]:


df.columns=df.columns.str.replace(' ','_')


# In[12]:


df=df.dropna(how='any')


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated()


# In[15]:


data = pd.merge(df,df1,on='Country_Code',how='left')


# In[16]:


data.head()


# In[17]:


data.info()


# # EDA
# 

# Explore the geographical distribution of the restaurants, finding out the
# cities with maximum / minimum number of restaurants.

# In[18]:


country_distri = data.groupby(['Country','Country_Code']).agg( Count = ('Restaurant_ID','count'))
df =country_distri.sort_values(by='Count',ascending=False)
df


# In[19]:


df.plot(kind='bar')


# In[20]:


vc = pd.DataFrame(data.Country.value_counts()).rename({'Country':'Frequency'},axis=1)
vc['Percentage'] = (vc.Frequency/vc.Frequency.sum()*100).round(2)


# In[21]:


sns.countplot(x='Country',data=data,order=vc.index)
sns.set(rc={"figure.figsize":(20, 5)})
# we can see that India has highest restaurants compared to other countries


# In[22]:


df.head(10).plot(kind='bar')


# In[23]:


city_distri = data.groupby(['City']).agg( Count = ('Restaurant_ID','count'))
city_distri1=city_distri.sort_values(by='Count',ascending=False)


# In[24]:


city_distri.max()
# City With Highest Restaurants In India Is New Delhi With 5473 Restaurants


# In[25]:


city_distri[city_distri['Count']==1].sum()


# In[26]:


city_distri.min()
# There is a total of 46 Cities with only single Restaurant.


# Restaurant franchising is a thriving venture. So, it is very important to explore the franchise with most national presence

# In[27]:


df = data.City.value_counts()[:5]
df.plot(kind='bar')
# As we can see New Delhi has the most number of restaurants.


# In[28]:


from wordcloud import WordCloud,STOPWORDS
import numpy as np
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=1440, height=1080, relative_scaling=0.5,stopwords=stopwords).generate_from_frequencies(data['Restaurant_Name'].value_counts())
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[29]:


rest = data['Restaurant_Name'].value_counts()[:10]
rest.plot(kind='bar')


# In[30]:


sns.barplot(data=rest,x=rest.values,y=rest.index).set(title='Restaurants Presence')


# the ratio between restaurants that allow table booking vs. those that do not allow table booking

# In[31]:


data1=data.copy()
data1.columns


# In[32]:


dummy = ['Has_Table_booking','Has_Online_delivery']
data1 = pd.get_dummies(data1,columns=dummy)
data1.head()


# In[33]:


table_b =data1[data1['Has_Table_booking_Yes']==1]['Restaurant_ID'].count()
table_bn =data1[data1['Has_Table_booking_Yes']==0]['Restaurant_ID'].count()
print('Ratio between Restaurant who allow table booking and who does not allow table booking:',round((table_b/table_bn),2))


# percentage of restaurants providing online delivery

# In[34]:


online_deliver = data1[data1['Has_Online_delivery_Yes']==1]['Restaurant_ID'].count()
restaurants = data1['Restaurant_ID'].count()
print('Percentage of Restaurants providing Online Delivery: ',(online_deliver/restaurants)*100)


# In[35]:


pd.crosstab(data['Has_Online_delivery'],data['Has_Table_booking'])


# difference in number of votes for the restaurants that deliver and the restaurants that do not deliver

# In[36]:


online_delivery = data1[data1['Has_Online_delivery_Yes']==1]['Votes'].sum()
No_delivery = data1[data1['Has_Online_delivery_Yes']==0]['Votes'].sum()


# In[37]:


print('Difference between Restaurants that deliver and Restaurants that do not deliver: ',(No_delivery-online_delivery))


#  top 10 cuisines served across cities
#  

# Maximum and minimum number of cuisines that a restaurant serves? Also, which is the most served cuisine across the restaurant for each city.

# In[38]:


# Top 10 Cuisines served 
Top_cuisines = data1.groupby(['City','Cuisines']).agg( Count = ('City','count'))
df = Top_cuisines.sort_values(by='Count',ascending=False)[:11]


# In[39]:


df.head(10).plot(kind='barh')


# In[40]:


cuisines_count = data1.groupby(['Restaurant_Name','Cuisines']).agg(Count=('Cuisines','count'))
cuisines_count.sort_values(by='Count',ascending=False)[:5]


# In[41]:


cuisines_count.sort_values(by='Count',ascending=True)[:5]


# Explore how ratings are distributed overall

# In[72]:


data['Rating_'] = data['Aggregate_rating'].round(0).astype(int)


# In[43]:


plt.figure(figsize=(15,4))
sns.countplot('Aggregate_rating',data=data[data.Aggregate_rating !=0])
plt.show()


# In[44]:


data['Rating_color'].value_counts()
color_represents=data.groupby(['Rating_color'],as_index=False)['Aggregate_rating'].mean()


# In[45]:


color_represents.columns=['Rating_color','Average_rating']


# In[46]:


color_represents =color_represents.sort_values(by='Average_rating',ascending=False)


# In[47]:


color_represents=color_represents[:5]
color_represents['Ratings']=['Excellent','Very Good','Good','Okay','Poor']


# In[48]:


color_represents


# distribution cost across the restaurants

# In[49]:


plt.figure(figsize=(15,5))
sns.distplot(data[data.Average_Cost_for_two !=0].Average_Cost_for_two)
plt.show()


# Explain the factors in the data that may have an effect on
# ratings e.g. No. of cuisines, cost, delivery option etc.

# In[57]:


data['Average_Cost_for_two_cat'] = pd.cut(data[data.Average_Cost_for_two !=0].Average_Cost_for_two,bins=[0,200,500,1000,3000,5000,10000,800000000],labels=['<=200','<=500','<=1000','<=3000','<=5000','<=10000','no limit'])


# In[65]:


ax = plt.subplot2grid((2,5), (0,0),colspan = 2)
sns.countplot(data['Average_Cost_for_two_cat']).set(title='Average Price')
ax = plt.subplot2grid((2,5), (0,0),colspan = 2)
ax = plt.subplot2grid((2,5), (0,2),colspan = 3)
sns.boxplot(x='Average_Cost_for_two_cat',y='Aggregate_rating',data=data)

count = data['Price_range'].value_counts().reset_index()
count.columns =['Price_range','Count']
ax = plt.subplot2grid((2,5), (1,0),colspan = 2)
sns.barplot(x='Price_range',y='Count',data =count).set(title='Price Range')
ax = plt.subplot2grid((2,5), (1,2),colspan = 3)
sns.boxplot(x='Price_range',y='Aggregate_rating',data=data)

plt.suptitle('Price Count & Rating Distribution', size = 30)
plt.show()


# In[86]:


ax = plt.subplot2grid((2,5), (0,0),colspan = 2)
sns.scatterplot(data=data,x='Aggregate_rating',y='Votes')
agg =data.pivot_table(index='Rating_',values='Votes',aggfunc='sum').reset_index()
agg['perc_votes']=(agg.Votes/agg.Votes.sum()*100)
agg


# As we can see nothing has much impact on the ratings except price.We can see feom the above data that ratings are decreasing with the increasing price while on the other hand Votes are increasing along with the ratings.

# In[ ]:




