from os import error
from time import sleep
from bs4 import BeautifulSoup
from numpy import string_
import numpy
import requests
import datetime
import pandas as pd

cols={"season","compettion_year", "city", "event", "compettor", "pos", "medal","sex", "noc", "year of birth","age", "cm", "kg"}
df=pd.DataFrame(columns=cols)
    
    
def main():
    createCsv()
    df.to_csv("res.csv")
    # at = pd.read_csv("res.csv")

   
domain = "http://www.olympedia.org"
    
def createCsv():
    url = "http://www.olympedia.org/editions/results"
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content, "html.parser")
    container = soup.find("div", attrs={"class": "container"})
    tables = container.findAll("table", attrs={"class": "table table-striped"})
    summerOlympic = tables[0]
    try:    
        findCompettionNameAndLink(summerOlympic, "summer")
    except Exception as e:
        print(e)

# first page
def findCompettionNameAndLink(olympicType, season):
    for tr in olympicType.findAll("tr"):
        td = tr.find("td")
        str1 = " "
        city = str1.join(td.text.split()[0:len(td.text.split()) -1])
        compettionYear = td.text.split()[len(td.text.split())-1]
        link = domain + td.find("a")['href']
        try:    
            getEvent(link, season, city, compettionYear)
        except Exception as e:
            print(e)
        

# second page

def getEvent(link, season, city, compettionYear):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    container = soup.find("div", attrs={"class": "container"})
    table = container.find("table", attrs={"class": "table"})
    trs = table.findAll("tr", attrs={"class": "odd"})
    for tr in trs: 
        a = tr.find("td").a
        eventName = a.text
        link = domain + a['href']
        try:    
            getEventDeatails(link, season, city, eventName, compettionYear)
        except Exception as e:
            print(e)
            
    trs = table.findAll("tr", attrs={"class": "even"})
    for tr in trs: 
        a = tr.find("td").a
        eventName = a.text
        newLink = domain + a['href']
        getEventDeatails(link, season, city, eventName,compettionYear)
      
# third page
def getEventDeatails(link, season, city, eventName,compettionYear):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    container = soup.find("div", attrs={"class": "container"})
    table = container.find("table", attrs={"class": "table table-striped"})
    trs = table.findAll("tr")
    numOfCols = len(trs[0].findAll("th"))
    print("numOfCols: ", numOfCols, " ,city: ", 
          city, " ,compettionYear: ", compettionYear,
          " ,eventName: " , eventName)
    
    if len(trs) > 0:
       posOfCompetitorName = findIndexOfCDetails(trs[0])
       print("posOfCompetitorName: ", posOfCompetitorName)
        #get rid of headline
        # trs = trs[1:]
       for tr in trs:
          tds = tr.findAll("td")
          if tds:
              a = tds[posOfCompetitorName].find("a")
              pos = tds[0].text

              if a:
                  compettorLink = domain + a['href']
                  compettorName = a.text

                  if pos == "1" or pos == "=1":
                    medal = "Gold"
                  elif  pos == "2" or pos == "=2":
                      medal = "Silver"
                  elif  pos == "3" or pos == "=3":
                      medal = "Bronze"
                  else:
                      medal = numpy.nan
                  try:    
                    getCompettorData(compettorLink, season, city, eventName, compettorName, pos, medal,compettionYear)
                    global df
                    df.to_csv("res.csv")


                  except Exception as e:
                      print(e)



def findIndexOfCDetails(tr):
    ths = tr.findAll("th")
    indexOfName = 0
    for i in range(0,len(ths)):
        if ths[i].text == "Competitor(s)" or ths[i].text == "Player" or ths[i].text == "Athlete" or ths[i].text == "Swimmer" or ths[i].text == "Judoka" or ths[i].text == "Gymnast":
            return i    
    return numpy.nan

def getCompettorData(link, season, city, eventName, compettorName, pos, medal,compettionYear):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    container = soup.find("div", attrs={"class": "container"})
    table = container.find("table",attrs={"class":"biodata"})
    trs = table.findAll("tr")
    # for unknown values
    sex = numpy.nan
    noc = numpy.nan
    cm = numpy.nan
    kg = numpy.nan

    yearOfBirth =numpy.nan
    age = numpy.nan
    for tr in trs:
        if tr.find("th").text == "Sex":
            sex = tr.find("td").text
        if tr.find("th").text == "NOC":
            noc = tr.find("td").text
        if tr.find("th").text == "Measurements":
            measurements = tr.find("td").text
            if "/" in measurements:
                cm = measurements.split("/")[0].split()[0]
                kg = measurements.split("/")[1].split()[0].split("-")[0]
        if tr.find("th").text == "Born":
            date = tr.find("td").text.split()[:3]
            if str.isdigit(date[len(date)-1]):
                yearOfBirth = date[len(date)-1]
                age = int(compettionYear) - int(yearOfBirth)
    newEntry = {'season': [season],'city': [city], 'compettion_year': [compettionYear],
    'event': [eventName], 'compettor': [compettorName],'pos': [pos],
    'medal': [medal], 'sex': [sex],'noc': [noc], 'year of birth': [yearOfBirth],
    'age': [age], 'cm': [cm], 'kg': [kg]}
    dataFrame2 = pd.DataFrame(newEntry,columns=cols)
    global df
    df = df.append(dataFrame2)
    sleep(1)
      
if __name__ == "__main__":
    
    main()