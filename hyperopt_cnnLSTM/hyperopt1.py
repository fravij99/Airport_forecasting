"""
Import all the basic stuff
"""

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from predlib import direction_to_angle, WindowGenerator, Model, RepeatBaseline

import matplotlib as mpl

#HPERPARAMETERS FUNCTIONS



mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


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


train_df.pop('Hour')
val_df.pop('Hour')
test_df.pop('Hour')
df.pop('Hour')

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


MAX_EPOCHS = 1
MAX_EVALS=50
input_hours=24
output_hours=1

OUT_STEPS = 12 * output_hours
IN_STEPS = 12 * input_hours

prediction_labels = ['Temperature']
num_features_predicted = len(prediction_labels)
multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df,
                               label_columns=prediction_labels)

# multi_window.plot()
params = {
              'layer_size':     hp.choice('layer_size', np.arange(30, 60, 3)),
              'model':          hp.choice('model', ['advanced_model']),
              'layer_number':   hp.choice('layer_number', np.arange(1, 3, 1)), 
              'lr':             hp.loguniform('lr', -10, -1)
              }

model1 = Model(multi_window, train_std['Temperature'], OUT_STEPS, num_features_predicted, MAX_EPOCHS)

#HYPERPARAMETER OPTIMIZATION
#Hyperparameter space: we want to optimize leraning rate and nodes number

trial=Trials()
best= model1.finding_best(trial, MAX_EVALS, params)

print(space_eval(params, best))

sns.set(style='darkgrid')
model1.plot_opt(trial)