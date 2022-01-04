from os import error
from bs4 import BeautifulSoup
from numpy import string_ as np
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


athletes=pd.read_csv('res.csv')


def EDA():
    #copy of athletes
    q1=athletes.copy()
    q1=q1.loc[:,["sex","compettion"]]
    q1.info()
    # Drop the None values
    q1 = q1.drop(q1.index[q1.sex == "None"])
    q1.sex.unique()
    # Num of females and males
    gender_counts=q1.sex.value_counts()
    gender_counts   
    # Graph 
    plt.figure(figsize=(12,6))
    plt.title('Gender')
    plt.pie(gender_counts,labels=gender_counts.index,autopct='%1.1f%%',startangle=180,shadow=True);
    #



    

def fun():
    athletes['age'].value_counts(normalize=True)
    print(athletes['sex'].value_counts().plot(kind='pie'))
