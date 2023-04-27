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
from plot_keras_history import plot_history, show_history
import matplotlib as mpl
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
#HPERPARAMETERS FUNCTIONS


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
sns.set(style='darkgrid')

"""
Let's get the features and the data split: 70% training, 20% validation, 10% test
"""

df=pd.read_csv('DatiBresso.csv')
n = len(df)
df=df[0:int(n*0.5)]

n = len(df)
column_indices = {name: i for i, name in enumerate(df.columns)}
print(df.head())

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

train_mean = df['Dew Point'].max()


train_std = train_df.std()

train_df =(train_df - train_df.min()) / (train_df.max() - train_df.min())
val_df = (val_df - val_df.min()) / (val_df.max() - val_df.min())
test_df = (test_df - test_df.min()) / (test_df.max() - test_df.min())


"""
This is the core the class WindowGenerator which essentially carries out all the boring stuff we need to do in
order to create an eligible piece if training-val-test data.
"""


MAX_EPOCHS = 3
MAX_EVALS=1
input_hours=12
output_hours=1

OUT_STEPS = 12 * output_hours
IN_STEPS = 12 * input_hours

prediction_labels = ['Dew Point']
num_features_predicted = len(prediction_labels)
multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df,
                               label_columns=prediction_labels)

#multi_window.plot()

#HYPERPARAMETER OPTIMIZATION
#Hyperparameter space: we want to optimize leraning rate and nodes number

params = {
              'layer_size':     (32),
              'model':          'advanced_model',
              'layer_number':   1,
              'conv_number':    3,
              'lr':             0.0001,
              'kernel':         6
              }

model1 = Model(multi_window, train_std['Dew Point'], OUT_STEPS, num_features_predicted, MAX_EPOCHS)

example=multi_window.example


trial=Trials()
#best= model1.finding_best(trial, MAX_EVALS, params, example)

#print(space_eval(params, best))


#model1.plot_opt(trial)

model1.advanced_model(OUT_STEPS, num_features_predicted, model_name='advanced_model', conv_number=params['conv_number'],
                              layer_number=params['layer_number'], layer_size=params['layer_size'])                   
history_advanced = model1.compile_and_fit(lr=params['lr'], CheckPoint=True,
                                                           epochs=MAX_EPOCHS,
                                                           patience=5)
            

plot_history(history_advanced)
plt.savefig('history.png')
multi_window.plot_renormalized(model=model1, train_mean=train_mean, train_std=train_std)