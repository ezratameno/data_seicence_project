#!/usr/bin/env python
# coding: utf-8

# In[1]:


#New New New


# In[2]:


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


# In[3]:


athletes=pd.read_csv('C:/Users/A/Downloads/Data Sience Project/res.csv')


# In[4]:


athletes.dtypes
# סוג המשתנים


# In[5]:


athletes.shape
#כמות נתונים


# In[6]:


athletes.head()
#Show Table


# In[7]:


medal_num={'Gold':1,'Silver':1,'Bronze':1}
sex_num={'Female':1,'Male':0}
athletes.medal.replace(medal_num,inplace=True)
athletes.sex.replace(sex_num,inplace=True)
athletes = athletes.iloc[: , 1:]
athletes.head()#Show Table
#Convert the medal and the sex column to numbers


# In[8]:


athletes.info()


# In[9]:


athletes.isnull().sum()


# In[10]:


athletes['pos'] = athletes['pos'].fillna(0)
athletes['medal'] = athletes['medal'].fillna(0)
athletes.head()
# Replacing all the NaN values to 0 in the pos column.


# In[11]:


athletes.duplicated().sum()
#sum the duplicated rows


# In[12]:


athletes.drop_duplicates()
# delete the double rows


# In[13]:


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


# In[14]:


sns.boxplot(athletes.age)
#גרף לחריגים בגילאים


# In[15]:


athletes.age.describe()
#מראה את הגיל המינימאלי מקסימלי וכו


# In[16]:


top_10_countries=athletes.noc.value_counts().sort_values(ascending=False).head(10)
top_10_countries
#עשר המדינות הטובות בעולם
#בכמות אנשים שהתחרו


# In[17]:


#plot for the top 10 countries-participans

plt.figure(figsize=(12,6))
plt.title('top 10 countries participans')
sns.barplot(x=top_10_countries.index,y=top_10_countries,palette='Set2');


# In[18]:


gender_counts=athletes.sex.value_counts()
gender_counts


# In[19]:


plt.figure(figsize=(12,6))
plt.title('Gender')
plt.pie(gender_counts,labels=gender_counts.index,autopct='%1.1f%%',startangle=180,shadow=True);


# In[20]:


plt.figure(figsize=(12,6))
plt.title("Athletes Age distribution")
plt.xlabel("Age")
plt.ylabel("Num of athletes")
plt.hist(athletes.age, bins=np.arange(10,80,2),color='orange',edgecolor='white')


# In[21]:


plt.figure(figsize=(20, 10))
a_medal=athletes.groupby('age')['medal'].count().reset_index()
sns.barplot(x='age',y='medal',data=a_medal)
plt.title('Medals by Age')
plt.xlim(0,50)
plt.xlabel('Age')
plt.ylabel('Medals')
plt.show()


# In[22]:


Contingent_Size=athletes.copy()
Contingent_Size['total_athletes']=''
Contingent_Size=Contingent_Size.loc[:,['compettion_year', 'noc','total_athletes']].groupby(['compettion_year', 'noc']).count().reset_index()
Contingent_Size=Contingent_Size.rename(columns={'noc':'Country Name','compettion_year':'Year'})
Contingent_Size.head()

#גודל משלחת לפי שנה ולפי מדינה

#add a grafh


# In[23]:


###########################
#athletes_=pd.read_csv('C:/Users/A/Downloads/Data Sience Project/test.csv')
#athletes_.dtypes
###########################
#Fix it after the new csv file.


# In[24]:


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


# In[25]:


pop = pd.read_csv('C:/Users/A/Downloads/Data Sience Project/pop.csv')
pop.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)

pop = pd.melt(pop, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'World Population')

pop['Year'] = pd.to_numeric(pop['Year'])
pop.head()


# In[26]:


w_pop = pd.read_csv('C:/Users/A/Downloads/Data Sience Project/womanPopulation.csv')
w_pop = w_pop.dropna()
w_pop.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)

w_pop = pd.melt(w_pop, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'Woman Population%')

w_pop['Year'] = pd.to_numeric(w_pop['Year'])

w_pop.head()


# In[27]:


gdp = pd.read_csv('C:/Users/A/Downloads/Data Sience Project/w_gdp.csv')

gdp.drop(['Indicator Name' , 'Indicator Code'], axis = 1, inplace = True)

gdp = pd.melt(gdp, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'gdp')

gdp['Year'] = pd.to_numeric(gdp['Year'])


gdp.head()


# In[28]:


athletes=athletes.rename(columns={'noc':'Country Name','compettion_year':'Year'})

print(athletes)


# In[29]:


athletes_merge=pop.merge(gdp, how='inner', on=['Country Name','Year','Country Code'])
athletes_merge=athletes_merge.merge(w_pop, how='inner', on=['Country Name','Year','Country Code'])

#athletes_merge=athletes_merge.merge(Contingent_Size, how='inner', on=['Country Name','Year'])


athletes_merge=athletes_merge.merge(athletes, how='inner', on=['Country Name','Year'])


athletes_merge.head()


# In[30]:


athletes_merge.isnull().sum()


# In[31]:


medal_per_country_per_year=athletes_merge.groupby(['Year','Country Name']).agg({'medal':'sum'}).reset_index()
medal_per_country_per_year.head()


# In[32]:


medal_per_country_per_year=medal_per_country_per_year.merge(Contingent_Size, how='inner', on=['Country Name','Year'])
medal_per_country_per_year=medal_per_country_per_year.merge(gdp, how='inner', on=['Country Name','Year'])
medal_per_country_per_year=medal_per_country_per_year.merge(pop, how='inner', on=['Country Name','Year','Country Code'])
medal_per_country_per_year.drop('Country Code', axis = 1, inplace = True)

medal_per_country_per_year.head()


# In[33]:


medal_per_country_per_year.dropna(how = 'any', inplace = True)

medal_per_country_per_year[medal_per_country_per_year['gdp'].apply(lambda x: str(x).isdigit())]

medal_per_country_per_year["gdp"] = medal_per_country_per_year["gdp"].astype("float") 

medal_per_country_per_year["total_athletes"] = medal_per_country_per_year["total_athletes"].astype("float") 

medal_per_country_per_year.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


###############################################
athletes1=athletes.drop(['cm','age','year of birth','season','pos','kg'], axis = 1)

athletes1=athletes1.merge(gdp, how='inner', on=['Country Name','Year'])
athletes1=athletes1.merge(pop, how='inner', on=['Country Name','Year','Country Code'])
athletes1.drop('Country Code', axis = 1, inplace = True)

athletes1=athletes1.dropna()

athletes1


# In[35]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


# In[36]:


categorical_features = ['city', 'medal', 'Country Name', 'compettor', 'Year', 'event', 'sex']
ordinal = OrdinalEncoder()
athletes1[categorical_features] = ordinal.fit_transform(athletes1[categorical_features])
athletes1
# משתנה קטג


# In[37]:


corr = athletes1.corr()
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corr, annot=True);


# In[61]:


#getting our X and y
#del athletes1['compettor']
#del athletes1['city']
#del athletes1['sex']


X = athletes1.drop('medal',axis=1)
y = athletes1['medal']


# In[62]:


from  sklearn.model_selection import train_test_split

#splitting X and y in training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


# In[63]:


X_train.shape, y_train.shape


# In[64]:


X_test.shape, y_test.shape


# In[65]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

numerical_columns = [col for col in athletes1.columns if ((athletes1.dtypes[col] != 'object') & (col not in ['medal','Year']))]

sc = StandardScaler()
X_train[numerical_columns] = sc.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = sc.transform(X_test[numerical_columns])

X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[66]:


y_pred = rf.predict(X_test)


# In[67]:


from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:\n\n", classification_report(y_test, y_pred))


# In[68]:


rf.score(X_test, y_test)


# In[69]:


y_predicted = rf.predict(X_test)


# In[70]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[72]:


from sklearn.metrics import plot_confusion_matrix


disp = plot_confusion_matrix(rf, X_test, y_test, 
                             cmap='Blues', values_format='d', 
                             display_labels=['0', '1'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:


# Create a random forest classifier
rf = rf.fit(X, y)
rf.score(X, y)

# Random Forests in sklearn will automatically calculate feature importance
importances = rf.feature_importances_
importances

# We can sort the features by their importance
sorted(zip(rf.feature_importances_, X), reverse=True)

# the decision tree and random forest have very similar values


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




