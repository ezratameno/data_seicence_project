#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install bar_chart_race')


# In[2]:


import pandas as pd
import bar_chart_race as bcr
import warnings
from IPython.display import Video
import bar_chart_race as bcr

warnings.filterwarnings("ignore")


# In[3]:


olympics_data=pd.read_csv('C:/Users/A/Downloads/Data Sience Project/res.csv')


# In[4]:


olympics_data = olympics_data[['compettion_year','medal','noc']].copy()
olympics_data['medal'].fillna(0, inplace = True)
olympics_data.dropna(axis = 0, inplace = True)


# In[5]:


medals_df = olympics_data[olympics_data['medal'] != 0]
medals_df = medals_df.pivot_table(columns = 'noc', index = 'compettion_year', values='medal', aggfunc = 'count')


# In[6]:


medals_df.fillna(0, inplace = True)


# In[7]:


medals_df


# In[8]:


medals_df = medals_df.cumsum()
medals_df


# In[9]:


bcr.bar_chart_race(df=medals_df,filename='./olympics.mp4',n_bars = 6, steps_per_period=30, sort='desc',period_length=2500,  title= "Medals in Olympics")


# In[10]:


Video('./olympics.mp4')

