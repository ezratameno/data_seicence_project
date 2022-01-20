#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import error
from bs4 import BeautifulSoup
from numpy import string_
import requests
import statsmodels.api as sm
import datetime
import pandas as p
from sklearn import metrics
from patsy import dmatrices
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
from scipy.stats import chi2_contingency
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


athletes=pd.read_csv('C:/Users/A/Downloads/Data Sience Project/res.csv')


# In[3]:


medal_num={'Gold':1,'Silver':1,'Bronze':1}
sex_num={'Female':1,'Male':0}
athletes.medal.replace(medal_num,inplace=True)
athletes.sex.replace(sex_num,inplace=True)
athletes = athletes.iloc[: , 1:]
athletes.head()#Show Table
#Convert the medal and the sex column to numbers


# In[4]:


athletes['pos'] = athletes['pos'].fillna(0)
athletes['medal'] = athletes['medal'].fillna(0)
athletes.head()
# Replacing all the NaN values to 0 in the pos column.


# In[5]:


athletes.drop_duplicates()
# delete the double rows


# In[6]:


#athletes = athletes.drop(athletes.index[athletes.age == "None"])
athletes=athletes.dropna()
athletes["age"] = athletes["age"].astype("int") 
athletes["medal"] = athletes["medal"].astype("int") 
athletes["year of birth"] = athletes["year of birth"].astype("int") 
athletes.head()
#athletes["medal"] = athletes["medal"].astype("int") 
#athletes["year of birth"] = athletes["year of birth"].astype("int") 
#athletes.dtypes
#Convert object to int (age, year of birth)


# In[7]:


sns.pairplot(athletes[['cm','kg','age']])


# In[8]:


# Giving each column a variable
#H = olympics_df['Height (m)']
#W = olympics_df['Weight']
#A = olympics_df['Age']
#B = olympics_df['BMI']
#M = olympics_df['Medal']


#plt.scatter(B,athletes['medal'],c="purple")
#plt.xlabel("BMI")
#plt.ylabel("Medal")
#plt.title("Medal by BMI")
#plt.savefig("static/img/Medal_BMI.png", bbox_inches='tight', pad_inches=0.5)


# In[ ]:





# In[9]:


# Using features with the highest importance
X = pd.get_dummies(athletes[["sex", "age", "cm", "kg","noc"]])
y = athletes["medal"]
print(X.shape, y.shape)


# In[10]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.3)


# In[15]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[16]:


print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")


# In[17]:


X = pd.get_dummies(athletes[["sex", "age", "cm", "kg","noc"]])
y = athletes["medal"].values.reshape(-1,1)
print(X.shape, y.shape)
feature_names = X


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




