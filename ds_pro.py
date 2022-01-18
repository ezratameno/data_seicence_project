#!/usr/bin/env python
# coding: utf-8

# In[1]:


#New New


# In[153]:


from os import error
from bs4 import BeautifulSoup
from numpy import string_
import requests
import datetime
import pandas as pd
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[154]:


athletes=pd.read_csv('C:/Users/A/Downloads/Data Sience Project/res_.csv')


# In[155]:


athletes.dtypes
# סוג המשתנים


# In[156]:


athletes.shape
#כמות נתונים


# In[157]:


athletes.head()
#Show Table


# In[158]:


medal_num={'Gold':1,'Silver':2,'Bronze':3}
sex_num={'Female':1,'Male':0}
athletes.medal.replace(medal_num,inplace=True)
athletes.sex.replace(sex_num,inplace=True)
athletes = athletes.iloc[: , 1:]
athletes.head()#Show Table
#Convert the medal and the sex column to numbers


# In[159]:


athletes.info()


# In[160]:


athletes.isnull().sum()


# In[161]:


athletes['pos'] = athletes['pos'].fillna(0)
athletes['medal'] = athletes['medal'].fillna(0)
athletes.head()
# Replacing all the NaN values to 0 in the pos column.


# In[162]:


#
# We hane NaN and None - check what we need to do with the drop and fillna!!!!!!!!!! 
#


# In[163]:


athletes.duplicated().sum()
#sum the duplicated rows


# In[164]:


athletes.drop_duplicates()
# delete the double rows


# In[165]:


#athletes = athletes.drop(athletes.index[athletes.age == "None"])

#athletes["age"] = athletes["age"].astype("int") 
#athletes["medal"] = athletes["medal"].astype("int") 
#athletes["year of birth"] = athletes["year of birth"].astype("int") 
#athletes.dtypes
#Convert object to int (age, year of birth)


# In[166]:


sns.boxplot(athletes.age)
#גרף לחריגים בגילאים


# In[167]:


athletes.age.describe()
#מראה את הגיל המינימאלי מקסימלי וכו


# In[168]:


top_10_countries=athletes.noc.value_counts().sort_values(ascending=False).head(10)
top_10_countries
#עשר המדינות הטובות בעולם


# In[169]:


#plot for the top 10 countries

plt.figure(figsize=(12,6))
plt.title('top 10 countries')
sns.barplot(x=top_10_countries.index,y=top_10_countries,palette='Set2');


# In[170]:


gender_counts=athletes.sex.value_counts()
gender_counts


# In[171]:


plt.figure(figsize=(12,6))
plt.title('Gender')
plt.pie(gender_counts,labels=gender_counts.index,autopct='%1.1f%%',startangle=180,shadow=True);


# In[172]:


plt.figure(figsize=(12,6))
plt.title("Athletes Age distribution")
plt.xlabel("Age")
plt.ylabel("Num of athletes")
plt.hist(athletes.age, bins=np.arange(10,80,2),color='orange',edgecolor='white')


# In[173]:


plt.figure(figsize=(20, 10))
a_medal=athletes.groupby('age')['medal'].count().reset_index()
sns.barplot(x='age',y='medal',data=a_medal)
plt.title('Medals by Age')
plt.xlim(0,50)
plt.xlabel('Age')
plt.ylabel('Medals')
plt.show()


# In[176]:


Contingent_Size=athletes.copy()
Contingent_Size['total_athletes']=''
Contingent_Size=Contingent_Size.loc[:,['compettion_year', 'noc','total_athletes']].groupby(['compettion_year', 'noc']).count().reset_index()
Contingent_Size=Contingent_Size.rename(columns={'noc':'Country Name','compettion_year':'Year'})
Contingent_Size.head()

#גודל משלחת לפי שנה ולפי מדינה

#add a grafh


# In[177]:


###########################
#athletes_=pd.read_csv('C:/Users/A/Downloads/Data Sience Project/test.csv')
#athletes_.dtypes
###########################
#Fix it after the new csv file.


# In[178]:


plt.figure(figsize=(12, 20))

plt.subplot(4, 2, 1)
athletes[athletes["sex"] == 1]["compettion_year"].hist(bins=35, color='blue', label='sex = f', alpha=0.6)
plt.legend()
plt.xlabel("compettion_year")

plt.subplot(4, 2, 2)
athletes[athletes["sex"] ==0]["compettion_year"].hist(bins=35, color='red', label='sex = m', alpha=0.6)
plt.legend()
plt.xlabel("compettion_year")

plt.subplot(4, 2, 3)
athletes[athletes["sex"] == 1]["compettion_year"].hist(bins=35, color='blue', label='sex = f', alpha=0.6)
athletes[athletes["sex"] == 0]["compettion_year"].hist(bins=35, color='red', label='sex = m', alpha=0.6)
plt.legend()
plt.xlabel("compettion_year")


#לשנות בהתאם לקובץ החדש 
#לדוגמא פיימייל ומייל צריכים להיות אפס ואחד וכו


# In[179]:


pop = pd.read_csv('C:/Users/A/Downloads/Data Sience Project/pop.csv')
pop.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)

pop = pd.melt(pop, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'World Population')

pop['Year'] = pd.to_numeric(pop['Year'])
pop.head()


# In[180]:


w_pop = pd.read_csv('C:/Users/A/Downloads/Data Sience Project/womanPopulation.csv')
w_pop = w_pop.dropna()
w_pop.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)

w_pop = pd.melt(w_pop, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'Woman Population%')

w_pop['Year'] = pd.to_numeric(w_pop['Year'])

w_pop.head()


# In[181]:


gdp = pd.read_csv('C:/Users/A/Downloads/Data Sience Project/w_gdp.csv')

gdp.drop(['Indicator Name' , 'Indicator Code'], axis = 1, inplace = True)

gdp = pd.melt(gdp, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'gdp')

gdp['Year'] = pd.to_numeric(gdp['Year'])


gdp.head()


# In[182]:


athletes=athletes.rename(columns={'noc':'Country Name','compettion_year':'Year'})

print(athletes_)


# In[ ]:





# In[ ]:





# In[ ]:





# In[184]:


athletes_merge=pop.merge(gdp, how='inner', on=['Country Name','Year','Country Code'])
athletes_merge=athletes_merge.merge(w_pop, how='inner', on=['Country Name','Year','Country Code'])

#athletes_merge=athletes_merge.merge(Contingent_Size, how='inner', on=['Country Name','Year'])


athletes_merge=athletes_merge.merge(athletes, how='inner', on=['Country Name','Year'])


athletes_merge.head(70)


# In[ ]:





# In[ ]:





# In[ ]:




