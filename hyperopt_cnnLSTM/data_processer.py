#!/usr/bin/env python
# -*- coding: utf-8 -*- #

"""
Import all the basic stuff
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

"""
For Ferenheit conversion
"""

from gemlib import to_celsius
from predlib import direction_to_angle, WindowGenerator, Model, RepeatBaseline

import IPython
import IPython.display
import matplotlib as mpl

"""
import tensorflow stuff
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from plot_keras_history import show_history, plot_history

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

"""
Automatically retrieves the path of the lastest `Bresso_updated` weather-data file downloaded through `table_scraper.py`
"""

gemini_path = '/Users/matteocitterio/Documents/gemini-Project'
files = [i for i in os.listdir(gemini_path) if os.path.isfile(os.path.join(gemini_path, i)) and
         'Bresso_updated' in i]

path_name = (files[-1])

"""
Read file and drop NaNs
"""

df = pd.read_csv(path_name)
df.dropna(how='any', inplace=True)

"""
Convert wind direction into degrees
"""

Written_winds = ['North', 'South', 'East', 'West']
Shorter_winds = ['N', 'S', 'E', 'W']

df.Wind = df.Wind.replace(Written_winds, Shorter_winds)

wind_dirs = ['N', 'NNE', 'NNW', 'NW', 'NE', 'ENE', 'E', 'SE',
             'ESE', 'SSE', 'WNW', 'W', 'S', 'SW', 'SSW', 'WSW']
angle_list = [0, 22.5, 337.5, 315, 45, 67.5, 90, 135,
              112.5, 157.5, 295.5, 270, 180, 225, 202.5, 247.5]

df.Wind = direction_to_angle(df.Wind, wind_dirs, angle_list)

"""
Stripe measure units and convert everything into float numbers
"""

head_list = [
    'Temperature',
    'Humidity',
    'Dew Point',
    'Speed',
    'Gust',
    'Pressure',
    'Precip. Rate.',
    'Precip. Accum.',
    'Solar',
]

for header in head_list:
    df[header] = (df[header].str.split().str[0])
    df[header] = (df[header].astype(float))

"""
Let's do the vector transformation (wind speed, wind direction) -> (wind x, wind y), this will improve our 
prediction
"""

wv = df.pop('Speed')

# Convert to radians.
wd_rad = df.pop('Wind')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
# plt.colorbar()
# plt.xlabel('Wind X [m/s]')
# plt.ylabel('Wind Y [m/s]')
# ax = plt.gca()
# ax.axis('tight')

# plt.show()

"""
Let's cut the complete dataset eliminating the parts where no data are available, namely consider only from
30-10-2021 until today
"""

LastAvailableDay = datetime.strptime(
    df['Time'].str.split().str[2].unique()[-1], "%Y-%m-%d")
StartingDay = LastAvailableDay - timedelta(days=375-1)
print(StartingDay)

df = df[df['Time'].str.split().str[2] >= StartingDay.strftime("%Y-%m-%d")]

df['Temperature'] = to_celsius(df['Temperature'])

"""
We can now think that time is of course periodic and an a priori assumption could be considering weather as
something with the same period of the time of the Year + the time of the day + fluctuations.
We should then convert the time info into a periodic variable
"""

date_time = pd.to_datetime(df.pop('Time'), format='%I:%M %p %Y-%m-%d')
timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day
hour = 60 * 60

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
df['Hour'] = (timestamp_s - min(timestamp_s)) / hour

# fig2, ax2 = plt.subplots(figsize=(10, 8))
# ax2.plot(np.array(df['Hour'])[:576],np.array(df['Day sin'])[:576])
# ax2.plot(np.array(df['Hour'])[:576],np.array(df['Day cos'])[:576])
# ax2.set_xlabel('Time [h]')
# ax2.set_title('Time of day signal')

# fig3, ax3 = plt.subplots(figsize=(10, 8))
# ax3 = sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='coolwarm', annot=True)

"""
Precipitation rate has only spikes and it is not as informative since it basically doesnt correlate with anything
we should drop it and keep in mind that the same fate could be shared by her sister, `Prep. Acc.`
"""

df.pop('Precip. Rate.')
df.head(5)

"""
Since i need to test what i'im doing and to stress my architecture features I only consider the 2% of the
complete dataset which is something around a week
"""

# n = len(df)
# df = df[int(n*0.95):-1]

df.to_csv('DatiBresso.csv')