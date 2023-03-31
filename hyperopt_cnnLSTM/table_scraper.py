#!/usr/bin/env python
# -*- coding: utf-8 -*- #

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime, timedelta

"""
This program will produce a `.csv` containing 90-days span wheater data of the Bresso Airport. 
The time span is computed starting from latest available data of the current day.

FUTURE UPDATES: program will simply modify (if present) the latest `.csv` file adding the new rows
of the current day and deleting rows related to 91-days ago or even older data.

"""

timespan=1000
today=datetime.today()
starting_day = today - timedelta(days = timespan+1)
print('Program will retrieve data from',starting_day.strftime("%Y-%m-%d"),'to today,',today.strftime("%Y-%m-%d"))

for single_date in (starting_day + timedelta(n) for n in range(timespan+1)):

    print('Doing: ', single_date.strftime("%Y-%m-%d"))

    # Create an URL object
    url = 'https://www.wunderground.com/dashboard/pws/IBRESS8/table/'+str(single_date.strftime("%Y-%m-%d"))+'/'+str(single_date.strftime("%Y-%m-%d"))+'/daily'
    # Create object page
    page = requests.get(url)
    #if this outputs "<Response[200]>" then we have the permission to fetch the website data
    print(page)

    # parser-lxml = Change html to Python friendly format
    # Obtain page's information
    soup = BeautifulSoup(page.text, 'lxml')

    # Obtain information from tag <table>
    table1 = soup.find("table", {'class': 'history-table desktop-table'})

    # Obtain every title of columns with tag <th>
    headers = []
    for i in table1.find_all('th'):
        title = i.text
        headers.append(title)

    # Create a dataframe
    mydata = pd.DataFrame(columns = headers)
    # Create a for loop to fill mydata
    for j in table1.find_all('tr')[2:]:
        row_data = j.find_all('td')
        row = [i.text for i in row_data]
        length = len(mydata)
        mydata.loc[length] = row

    #Doing a bit of cleening 
    mydata['Time'] = mydata['Time'].astype(str) + ' '+ str(single_date.strftime("%Y-%m-%d"))
    mydata['Humidity']=mydata["Humidity"].str.replace("°","")     
    mydata['Speed']=mydata["Speed"].str.replace("°","")
    mydata['Gust']=mydata["Gust"].str.replace("°","")
    mydata['Pressure']=mydata["Pressure"].str.replace("°","")
    mydata['Precip. Rate.']=mydata["Precip. Rate."].str.replace("°","")
    mydata['Precip. Accum.']=mydata["Precip. Accum."].str.replace("°","")

    # if file does not exist write header 
    if not os.path.isfile('Bresso_updated_'+str(today.strftime("%Y-%m-%d %H-%M"))+'.csv'):
        mydata.to_csv('Bresso_updated_'+str(today.strftime("%Y-%m-%d %H-%M"))+'.csv', index=False)
    else: # else it exists so append without writing the header
        mydata.to_csv('Bresso_updated_'+str(today.strftime("%Y-%m-%d %H-%M"))+'.csv', mode='a', index=False, header=False)

    print('Done with: ', str(single_date.strftime("%Y-%m-%d")))

df = pd.read_csv('Bresso_updated_'+str(today.strftime("%Y-%m-%d %H-%M"))+'.csv')
print(df)