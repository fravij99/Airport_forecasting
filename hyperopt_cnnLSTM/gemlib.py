#!/usr/bin/env python
# -*- coding: utf-8 -*- #

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import time
from sklearn.metrics import mean_squared_error
import seaborn as sns
import os
from datetime import datetime, timedelta

def to_celsius(temp):

    """
    Converts ferenheit to celsius
    """

    return (temp - 32) * 5/9

# Loading dataset
def load_data(path_name):
    """
    This function loads data into a pd.DataFrame and stripes measure units. Additionally,
    it converts into [0,1] numbers and drops some columns.
    #Params:
        -) `path_name` : `str` - location of the file
    #Returns: 
        -) `df` : `Pandas.DataFrame` - the loaded and striped data
        -) `max_temp` : `double` - maximum observed temperature used for data normalization
    """

    df = pd.read_csv(path_name)

    # this stripes out all the measure units
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

    df.drop(columns=['Wind'], axis=1, inplace=True)
    df.dropna(how='any', inplace=True)
    max_temp = 0

# Deleating units in dataset using str.plit function

    for header in head_list:
        df[header] = (df[header].str.split().str[0])
        if (header == 'Temperature'):
            max_temp = df[header].max()
        df[header] = (df[header].astype(float))
        df[header] = df[header]/df[header].max()

    return df, float(max_temp)

def CorrMatrix(data):
    """
    Plots an `sns.heatmap` which represents the correlation matrix
    #Params:
        -)`data` : `Pandas.DataFrame` - the data used for building the correlation matrix
    """

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(data.corr(), vmin=-1, vmax=1,
                     cmap="coolwarm",
                     ax=ax,
                     annot=True)
    plt.show()

def GetADataPortion(data, portion):
    """
    This function returns a new dataframe containing only data related to the last week (the lastest 7 days)
    #Params:
        -) `data` : `Pandas.Dataframe` - The data you want to reduce
        -) `portion` : `int` - Number of days you want to reduce your dataset to
    #Returns: 
        -) `ReducedData` : `Pandas.DataFrame` - The reduced dataframe
    """

    LastAvailableDay = datetime.strptime(
        data['Time'].str.split().str[2].unique()[-1], "%Y-%m-%d")
    StartingDay = LastAvailableDay - timedelta(days=portion-1)

    ReducedData = data[data['Time'].str.split().str[2] >= StartingDay.strftime("%Y-%m-%d")]
    ReducedData = ReducedData.reset_index()
    ReducedData.drop(columns=['index'], inplace=True)

    return ReducedData


def SlidingWindow(df, HourLag, TrainingWindowLenght, ValidationWindowLenght, TestWindowLenght):
    """
    Executes the sliding of data returning the maximum amount of windows given the dataframe
    #PARAMS:
        -) `df` : `Pandas.Dataframe` - The data you want to slice
        -) `HourLag` : `int` -  number of points corrisponding of the our lag. E.g for an hour lag we need 12 
                                points
        -) `TrainingWindowLenght` - `int` - number of points for each training window
        -) `ValidationWindowLenght` - `int` - number of points for each val window
        -) `TestWindowLenght` - `int` - number of points for each test window

    #RETURNS:

        -) X and Y arrays as traninig-val-test data
        -) `Number of created windows` - `int` - Number of generated windows

    """

    XTrainingSet, XValidationSet, XTestSet = [], [], []
    YTrainingSet, YValidationSet, YTestSet = [], [], []

    StartingIndex = len(df) - ValidationWindowLenght - TestWindowLenght
    HeadersNoTempTime = [x for x in df.columns if not x in ['Temperature','Time']]
    print('Headers used for training: ',HeadersNoTempTime)

    while (StartingIndex - TrainingWindowLenght >= 0):

        YTrainingSet.append(
            df['Temperature'][StartingIndex-TrainingWindowLenght:StartingIndex])
        YValidationSet.append(
            df['Temperature'][StartingIndex:StartingIndex+ValidationWindowLenght])
        YTestSet.append(df['Temperature']
                        [StartingIndex+ValidationWindowLenght:])

        XTrainingSet.append(
            df[HeadersNoTempTime][StartingIndex-TrainingWindowLenght:StartingIndex])
        XValidationSet.append(
            df[HeadersNoTempTime][StartingIndex:StartingIndex+ValidationWindowLenght])
        XTestSet.append(df[HeadersNoTempTime][StartingIndex+ValidationWindowLenght:])

        df.drop(df.tail(HourLag).index, inplace=True)
        StartingIndex = StartingIndex-HourLag


    XValidationSet = np.asarray(XValidationSet)
    XTrainingSet = np.asarray(XTrainingSet)
    XTestSet = np.asarray(XTestSet)
    YTrainingSet = np.asarray(YTrainingSet)
    YValidationSet = np.asarray(YValidationSet)
    YTestSet = np.asarray(YTestSet)

    NumberOfCreatedWindows = XValidationSet.shape[0]

    XTrainingSet=XTrainingSet.reshape(TrainingWindowLenght, NumberOfCreatedWindows, 9)
    YTrainingSet=YTrainingSet.reshape(TrainingWindowLenght, NumberOfCreatedWindows, 1)
    XValidationSet=XValidationSet.reshape(ValidationWindowLenght, NumberOfCreatedWindows, 9)
    YValidationSet=YValidationSet.reshape(ValidationWindowLenght, NumberOfCreatedWindows, 1)
    XTestSet=XTestSet.reshape(TestWindowLenght, NumberOfCreatedWindows, 9)
    YTestSet=YTestSet.reshape(TestWindowLenght,NumberOfCreatedWindows,1)

    return XTrainingSet, XValidationSet, XTestSet, YTrainingSet, YValidationSet, YTestSet , NumberOfCreatedWindows
    

def LSTM_model_tot(input_shape,activation='tanh',dropout=0.2,out_steps=24,num_features=9):
    """
    Builds an LSTM.keras.model with encoder(?)
    #params:
    -) activation [str]: activation function, default is tanh
    -) input_shape [tuple]: input shape
    -) droput [double]: dropout parameter, default is 0.2

    return: it returns the model
    """

    print(input_shape)

    enco_deco=tf.keras.models.Sequential()
    enco_deco.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(75, activation=activation, return_sequences=True), input_shape=input_shape))
    enco_deco.add(tf.keras.Dropout(dropout))
    enco_deco.add(tf.keras.layers.Dense(out_steps*num_features,
                                        kernel_initializer=tf.initializers.zeros()))
    enco_deco.add(tf.keras.layers.Reshape([out_steps, num_features]))

    return enco_deco