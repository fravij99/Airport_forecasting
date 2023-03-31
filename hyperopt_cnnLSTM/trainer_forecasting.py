#!/usr/bin/env python
# -*- coding: utf-8 -*- #

"""
Import all the basic stuff
"""
import hyperopt as hp
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

"""
For Ferenheit conversion
"""

from gemlib import to_celsius
from predlib import direction_to_angle, WindowGenerator, Model, RepeatBaseline

import matplotlib as mpl

"""
import tensorflow stuff
"""

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# """
# Automatically retrieves the path of the lastest `Bresso_updated` weather-data file downloaded through `table_scraper.py`
# """

# gemini_path = '/Users/matteocitterio/Documents/gemini-Project'
# files = [i for i in os.listdir(gemini_path) if os.path.isfile(os.path.join(gemini_path, i)) and
#          'Bresso_updated' in i]

# path_name = (files[-1])

# """
# Read file and drop NaNs
# """

# df = pd.read_csv(path_name)
# df.dropna(how='any', inplace=True)

# """
# Convert wind direction into degrees
# """

# Written_winds = ['North', 'South', 'East', 'West']
# Shorter_winds = ['N', 'S', 'E', 'W']

# df.Wind = df.Wind.replace(Written_winds, Shorter_winds)

# wind_dirs=['N', 'NNE', 'NNW', 'NW', 'NE', 'ENE', 'E', 'SE', 'ESE','SSE', 'WNW', 'W', 'S', 'SW', 'SSW', 'WSW']
# angle_list=[0,22.5,337.5,315,45,67.5,90,135,112.5,157.5,295.5,270,180,225,202.5,247.5]

# df.Wind=direction_to_angle(df.Wind,wind_dirs,angle_list)

# """
# Stripe measure units and convert everything into float numbers
# """

# head_list = [
#     'Temperature',
#     'Humidity',
#     'Dew Point',
#     'Speed',
#     'Gust',
#     'Pressure',
#     'Precip. Rate.',
#     'Precip. Accum.',
#     'Solar',
# ]

# for header in head_list:
#         df[header] = (df[header].str.split().str[0])
#         df[header] = (df[header].astype(float))

# """
# Let's do the vector transformation (wind speed, wind direction) -> (wind x, wind y), this will improve our
# prediction
# """

# wv = df.pop('Speed')

# # Convert to radians.
# wd_rad = df.pop('Wind')*np.pi / 180

# # Calculate the wind x and y components.
# df['Wx'] = wv*np.cos(wd_rad)
# df['Wy'] = wv*np.sin(wd_rad)

# # plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
# # plt.colorbar()
# # plt.xlabel('Wind X [m/s]')
# # plt.ylabel('Wind Y [m/s]')
# # ax = plt.gca()
# # ax.axis('tight')

# # plt.show()

# """
# Let's cut the complete dataset eliminating the parts where no data are available, namely consider only from
# 30-10-2021 until today
# """

# LastAvailableDay = datetime.strptime(
#         df['Time'].str.split().str[2].unique()[-1], "%Y-%m-%d")
# StartingDay = LastAvailableDay - timedelta(days=375-1)
# print(StartingDay)

# df = df[df['Time'].str.split().str[2] >= StartingDay.strftime("%Y-%m-%d")]

# df['Temperature'] = to_celsius(df['Temperature'])

# """
# We can now think that time is of course periodic and an a priori assumption could be considering weather as
# something with the same period of the time of the Year + the time of the day + fluctuations.
# We should then convert the time info into a periodic variable
# """

# date_time = pd.to_datetime(df.pop('Time'), format='%I:%M %p %Y-%m-%d')
# timestamp_s=date_time.map(pd.Timestamp.timestamp)

# day = 24*60*60
# year = (365.2425)*day
# hour = 60 * 60

# df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
# df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
# df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
# df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
# df['Hour'] = (timestamp_s - min(timestamp_s)) / hour

# # fig2, ax2 = plt.subplots(figsize=(10, 8))
# # ax2.plot(np.array(df['Hour'])[:576],np.array(df['Day sin'])[:576])
# # ax2.plot(np.array(df['Hour'])[:576],np.array(df['Day cos'])[:576])
# # ax2.set_xlabel('Time [h]')
# # ax2.set_title('Time of day signal')

# # fig3, ax3 = plt.subplots(figsize=(10, 8))
# # ax3 = sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='coolwarm', annot=True)

# """
# Precipitation rate has only spikes and it is not as informative since it basically doesnt correlate with anything
# we should drop it and keep in mind that the same fate could be shared by her sister, `Prep. Acc.`
# """

# df.pop('Precip. Rate.')
# df.head(5)

# """
# Since i need to test what i'im doing and to stress my architecture features I only consider the 2% of the
# complete dataset which is something around a week
# """

# # n = len(df)
# # df = df[int(n*0.95):-1]

"""
Let's get the features and the data split: 70% training, 20% validation, 10% test
"""

df=pd.read_csv('DatiBresso.csv')

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

Hours = df['Hour']
Days = Hours/24

fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.plot(df['Hour']/24, df['Temperature'])
ax4.set_xlabel('Days')
ax4.set_ylabel('Temperature[°C]')
ax4.axvspan(xmin=train_df['Hour'].iloc[0]/24,
            xmax=train_df['Hour'].iloc[-1]/24,alpha=0.1, color='red', label='train')
ax4.axvspan(xmin=val_df['Hour'].iloc[0]/24,
            xmax=val_df['Hour'].iloc[-1]/24, alpha=0.1, color='green', label='val')
ax4.axvspan(xmin=test_df['Hour'].iloc[0]/24,
            xmax=test_df['Hour'].iloc[-1]/24, alpha=0.1, color='blue', label='test')
ax4.legend()
ax4.set_title('Temperature Dataset Split')


train_df.pop('Hour')
val_df.pop('Hour')
test_df.pop('Hour')
df.pop('Hour')

plt.show()

num_features = df.shape[1]
print('Num features:', num_features)

"""
Let's renormalize the dataset according to the train mean and std
"""

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


"""
This is the core the class WindowGenerator which essentially carries out all the boring stuff we need to do in
order to create an eligible piece if training-val-test data.
"""
params={
              'layer_size':     ('layer_size', 6),
              'model':          ('model', "dense"),
              'layer_number':   ('batch_size', 3)
              }

MAX_EPOCHS = 20

input_hours=24
output_hours=1

OUT_STEPS = 12 * output_hours
IN_STEPS = 12 * input_hours

prediction_labels=['Temperature']
num_features_predicted = len(prediction_labels)
multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df,
                               label_columns=prediction_labels)

# multi_window.plot()

model = Model(multi_window, train_std['Temperature'], params, OUT_STEPS, num_features_predicted)

train_lstm = True
train_bidirectional_lstm = True
train_advanced_model = False
train_repeat_baseline = False
train_cnn = True
train_dense = True
train_multilinear = True

if train_lstm:

    """
    Define and train LSTM model
    """

    model.multi_lstm(OUT_STEPS, num_features_predicted, model_name='multi_lstm')
    history_lstm = model.compile_and_fit(CheckPoint=True,
                                         epochs=MAX_EPOCHS,
                                         patience=5)
    # plot_history(history_lstm)
    # multi_window.plot_renormalized(model=model, train_mean=train_mean, train_std=train_std)

if train_bidirectional_lstm:

    """
    Define and train BIDIRECTIONAL LSTM model
    """

    model.bidirectional_lstm(OUT_STEPS, num_features_predicted, model_name='bidirectional_lstm')
    history_birirectional_lstm = model.compile_and_fit(CheckPoint=True, epochs=MAX_EPOCHS,patience=5)
    # plot_history(history_birirectional_lstm)
    # multi_window.plot_renormalized(model=model, train_mean=train_mean, train_std=train_std)

if train_advanced_model:

    """
    Define and train  ADVANCED model
    """

    model.advanced_model(OUT_STEPS, num_features_predicted, model_name='advanced_model')
    history_advance_model = model.compile_and_fit(CheckPoint=True, epochs=MAX_EPOCHS,patience=5)
    #plot_history(history_advance_model)
    multi_window.plot_renormalized(model=model, train_mean=train_mean, train_std=train_std)

if train_repeat_baseline:

    model.repeat_baseline(OUT_STEPS, num_features_predicted, model_name='repeat_baseline')
    history_repeat_baseline = model.compile_and_fit(epochs=MAX_EPOCHS,
                                                    patience=5)

if train_multilinear:

    """
    Define and train MULTILINEAR model
    """
    model.multi_linear(OUT_STEPS, num_features_predicted, model_name='multilinear')
    history_multilinear = model.compile_and_fit(CheckPoint=True,
                                                epochs=MAX_EPOCHS,
                                                patience=5)
    # plot_history(history_multilinear)
    # multi_window.plot_renormalized(
    #     model=model, train_mean=train_mean, train_std=train_std)

if train_dense:

    """
    Define and train DENSE model
    """

    model.dense(OUT_STEPS, num_features_predicted, model_name='dense')
    history_dense = model.compile_and_fit(CheckPoint=True,
                                          epochs=MAX_EPOCHS,
                                          patience=5)
    # plot_history(history_dense)
    # multi_window.plot_renormalized(
    #     model=model, train_mean=train_mean, train_std=train_std)

if train_cnn:

    """
    Define and train CNN model
    """

    model.cnn(OUT_STEPS, num_features_predicted, model_name='cnn')
    history_cnn = model.compile_and_fit(CheckPoint=True,
                                        epochs=MAX_EPOCHS,
                                        patience=5)
    # plot_history(history_cnn)
    # multi_window.plot_renormalized(
    #     model=model, train_mean=train_mean, train_std=train_std)

"""
Plot performances_comparison
"""

model.performance()
model.comparison_performances()




