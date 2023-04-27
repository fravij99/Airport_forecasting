#!/usr/bin/env python
# -*- coding: utf-8 -*- #

import numpy as np
import pandas as pd
from sklearn.base import OutlierMixin
import tensorflow as tf
import os
from datetime import datetime
import random
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK

import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from plot_keras_history import plot_history
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def direction_to_angle(direction, wind_dirs, angle_list):
    """
    It takes a direction (str) and outputs a direction (float) in terms of degrees
    """
    list_wind = []
    for i in range(0, len(direction)):
        index = wind_dirs.index(direction.iloc[i])
        list_wind.append(angle_list[index])
    return list_wind


class WindowGenerator:

    """
    The constructor of this class takes in input the window width, the shift and the data and produces as
    an output the right set of indexis for a desired window
    """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None,batch_size=32):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size=batch_size
        self.input_width=input_width

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

        """
        Let's make a simple example. Consider that your data are collected hourly so that you have a point of
        your series every hour. Let's say you want to consider as `input` of your model a sequence of 24 hours 
        and you want to predict the next two hours, so that you have as label a sequence of 2 points. In total
        (if you don't want your input sequence to be overlapped by labels, put shift=inputs_len) you'd have some-
        thing like:

        --Python:

        `INPUT_STEPS = 24
        OUT_STEPS = 2 # = len(labels)
        Window_example = WindowGenerator(input_width=INPUT_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)`

        --Return:

        `Total window size: 26
        Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
        Label indices: [24 25]
        Label column name(s): None`

        Where, if not specified, the  default Label column name(s) is 'Temperature'

        """

    def __repr__(self):
        """
        This simply prints out what follows whenever a new window is built
        """

        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        """
        This effectively splits the data (with all the related features) into the desired window shape (inputs,
        labels)
        """

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        """ 
        The `None` shape simply means that it could be anything, it is not fixed.
        """
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Temperature', max_subplots=3):
        """
        Useful function that plot the different quantities on an example window of data. The example
        retrieved throgh `self.example` method is random.
        """

        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data, shuffle=True, stride=1):
        """
        This function makes a keras dataset diveded into batches.
        """

        data = np.array(data, dtype=np.float32)
        print(self.total_window_size, len(data))
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=stride,
            shuffle=shuffle,
            batch_size=8,)

        ds = ds.map(self.split_window)

        return ds


    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def unshuffled_test(self):
        return self.make_dataset(self.test_df, shuffle=False, stride=self.label_width)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = None
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            randomator = tf.random.uniform(
                shape=[], maxval=self.batch_size, dtype=tf.int32, seed=random.seed(datetime.now()))
            it = iter(self.unshuffled_test)
            for i in range(0, randomator):
                result = next(it)

            print('random number: ', randomator)
            # And cache it for next time
            # self._example = result

        return result

    def plot_renormalized(self, train_mean, train_std, model, plot_col='Dew Point', max_subplots=3):

        a = list(np.arange(5, self.label_width, 5))
        str_list = ['+'+str(i) for i in a]
        inputs, labels = self.example

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [°C]')
            # plt.plot(self.input_indices, (inputs[n, :, plot_col_index]+train_mean)*train_std)
            # label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue


            plt.plot(self.label_indices, ((
                labels[n]*train_mean)), '-', c='#2ca02c')
            plt.scatter(self.label_indices, ((
                labels[n]*train_mean)), edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:

                predictions = model.predict(inputs)

                pred2=tf.reshape(predictions[n], self.label_width)

            #if (np.array(predictions[n+1, :, label_col_index]== predictions[n+2, :, label_col_index])).all:
             #   message = "First value and second value are equal!"
              #  print(message)
               # return -1


            plt.plot(self.label_indices, ((
                pred2*train_mean)), '-', c='#ff7f0e', ms=1)
            plt.scatter(self.label_indices, ((pred2*train_mean)), marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

            #plt.xticks(ticks=list(np.arange(self.input_width, self.input_width+self.label_width)), labels=str_list)
            plt.title(model.model_name)

        plt.xlabel('minutes into the future')
        plt.savefig('pred.png')
        plt.show()
        print('difference averege for every window in °C: ', np.absolute(np.array(predictions[0]-labels[0])).mean()*train_mean,
                                                             np.absolute(np.array(predictions[1]-labels[1])).mean()*train_mean,
                                                             np.absolute(np.array(predictions[2]-labels[2])).mean()*train_mean)
        

                    


    def xcross(self, vec1, vec2, index):
        cross = scipy.signal.correlate(vec1, vec2)
        plt.scatter(np.arange(len(cross))*5, cross, label='xcorr', c='lightblue', marker='X', edgecolors='k')
        plt.title('cross correlation ')
        plt.show()
        print(cross)


        


class Model():


    def __init__(self, window, resclale_factor, OUT_STEPS, num_features_predicted, max_epochs):

        self.max_epochs=max_epochs
        self.multi_val_performance = {}
        self.multi_performance = {}
        self.window = window
        self.rescale = resclale_factor
        self.trials = {}
        self.OUT_STEPS = OUT_STEPS
        self.num_features_predicted = num_features_predicted
        self.example={}

    def linear_lstm(self, OUT_STEPS, num_features_predicted, model_name, layer_number, layer_size, LoadModel=False):

        model = tf.keras.Sequential()
        for i in range(layer_number):
            model.add((tf.keras.layers.GRU(units = layer_size, activation='tanh', return_sequences=False)))
            model.add(tf.keras.layers.Dropout(0.01))
            model.add(tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()))
            # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])

        self.model_name = model_name
        self.model = model

        if LoadModel:

            self.load_pretrained_model()

    def double(self, OUT_STEPS, num_features_predicted, model_name, LoadModel=False):

        model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(
                64, return_sequences=True, activation='tanh'),
            tf.keras.layers.LSTM(
                32, return_sequences=False, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])
        ])

        self.model_name = model_name
        self.model = model

        if LoadModel:

            self.load_pretrained_model()

    def bidirectional_lstm(self, OUT_STEPS, num_features_predicted, model_name, layer_number, layer_size, LoadModel=False):

        model = tf.keras.Sequential()
        for i in range(layer_number):
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = layer_size, activation='tanh', return_sequences=False)))
            model.add(tf.keras.layers.Dropout(0.01))
            model.add(tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()))
            # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])



        self.model = model
        self.model_name = model_name


        if LoadModel:

            self.load_pretrained_model()

    def advanced_model(self, OUT_STEPS, num_features_predicted, model_name, conv_number, layer_number, layer_size, LoadModel=False):

        model = tf.keras.models.Sequential()


        for i in range(conv_number):
            model.add(tf.keras.layers.Conv1D(int(64/(2**i)), kernel_size=int(6), activation='relu'))
            model.add(tf.keras.layers.MaxPooling1D(2))
            #model.add(tf.keras.layers.Dropout(0.2))
        for i in range(layer_number):
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(layer_size/(2**i)), activation='tanh',
                                 return_sequences=True)))
            #model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())
            
        model.add(tf.keras.layers.Flatten())
        #model.add(tf.keras.layers.Dropout(0.2))

        #model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(OUT_STEPS*num_features_predicted))
        model.add(tf.keras.layers.Reshape([OUT_STEPS,num_features_predicted]))

        

        self.model = model
        self.model_name = model_name

        if LoadModel:

            self.load_pretrained_model()

    def convlstm1d(self, OUT_STEPS, num_features_predicted, model_name, layer_number, layer_size, LoadModel=False):

        model = tf.keras.Sequential()
        

        model.add(tf.keras.layers.ConvLSTM1D(16, 6, activation='tanh', recurrent_activation='hard_sigmoid', 
                                kernel_initializer=tf.initializers.zeros(), recurrent_dropout=0.02))

        model.add(tf.keras.layers.Dense(OUT_STEPS*num_features_predicted, activation='tanh', 
                                kernel_initializer=tf.initializers.zeros()))
        model.add(tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted]))


        self.model = model
        self.model_name = model_name        

    def multi_linear(self, OUT_STEPS, num_features_predicted, model_name, layer_number, layer_size, LoadModel=False):

        model = tf.keras.Sequential()
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        model.add(tf.keras.layers.Lambda(lambda x: x[:, -1:, :]))
        # Shape => [batch, 1, out_steps*features]
        for i in range((layer_number)):
            model.add(tf.keras.layers.Dense(OUT_STEPS*num_features_predicted, activation='tanh', 
                                kernel_initializer=tf.initializers.zeros()))
           # model.add(tf.keras.layers.Dropout(0.2))
        # Shape => [batch, out_steps, features]
        model.add(tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted]))


        self.model = model
        self.model_name = model_name

        if LoadModel:

            self.load_pretrained_model()

    def dense(self, OUT_STEPS, num_features_predicted, model_name, layer_number, layer_size, LoadModel=False):

        model = tf.keras.Sequential()

            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
        for i in range(layer_number):
            model.add(tf.keras.layers.Lambda(lambda x: x[:, -1:, :]))
            # Shape => [batch, 1, dense_units]

        
            model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            # Shape => [batch, out_steps*features]
            model.add(tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros(), activation='relu'))

            # Shape => [batch, out_steps, features]
        model.add(tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted]))


        self.model = model
        self.model_name = model_name

        if LoadModel:

            self.load_pretrained_model()

    def cnn(self, OUT_STEPS, num_features_predicted, model_name, kernel, layer_size, LoadModel=False):

        CONV_WIDTH = 6

        model = tf.keras.Sequential()
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        model.add(tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]))
            # Shape => [batch, 1, conv_units]

        model.add(tf.keras.layers.Conv1D(
                int(layer_size), activation='relu', kernel_size=(int(kernel))))
            # Shape => [batch, 1,  out_steps*features]
        model.add(tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()))
            # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])


        self.model = model
        self.model_name = model_name

        if LoadModel:

            self.load_pretrained_model()

    def repeat_baseline(self, model_name):

        self.model = RepeatBaseline()
        self.model_name = model_name

    def compile_and_fit(self, lr, EarlyStopping=True, TensorBoard=True, CheckPoint=False, epochs=20, patience=5):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        checkpoint_path = f"{self.model_name}/cp_.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./log')

        print('\n\n')
        print('Model name', self.model_name)

        if self.model_name == 'repeat_baseline':
            print('\nRepeat beaseline model, different compilation:\n')
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                               metrics=[tf.keras.metrics.MeanAbsoluteError()])

            history = None

        else:

            self.compile(lr)

            callbacks = []

            if (CheckPoint):
                callbacks.append(cp_callback)
                print('Saving weights')
            if (EarlyStopping):
                callbacks.append(early_stopping)
                print('Doing early stopping with patience =', patience)
            if (TensorBoard):
                callbacks.append(tensorboard_callback)
                print('Tensorboard callback available')

            print('\n')

            history = self.model.fit(self.window.train, epochs=epochs,
                                     validation_data=self.window.val,
                                     callbacks=callbacks, batch_size=32)

        self.multi_val_performance[self.model_name] = self.model.evaluate(self.window.val)
        self.multi_performance[self.model_name] = self.model.evaluate(self.window.test, verbose=0)

        return history

    def compile(self, lr):

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])


    def comparison_performances(self):

        x = np.arange(len(self.multi_performance))
        width = 0.3

        metric_name = 'mean_absolute_error'
        metric_index = self.model.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in self.multi_val_performance.values()]
        test_mae = [v[metric_index] for v in self.multi_performance.values()]

        plt.figure(figsize=(9, 7))
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=self.multi_performance.keys(),rotation=45)
        plt.ylabel(f'MAE [°C] (average over all times and outputs)')
        _ = plt.legend()

        print(self.multi_val_performance)
        print(self.multi_performance)

        plt.show()

    def performance(self):

        self.multi_val_performance[self.model_name] = np.asarray(self.model.evaluate(self.window.val)) * self.rescale
        self.multi_performance[self.model_name] = np.asarray(self.model.evaluate(self.window.test, verbose=0)) * self.rescale

    def predict(self, inputs):
        return self.model(inputs)

    def load_pretrained_model(self):

        checkpoint_path = f"{self.model_name}/cp_.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        self.model.load_weights(latest)
        self.compile()

    def optimize_hyperparams(self, MAX_EPOCHS, parameters):
        
        if parameters['model'] == 'cnn':


            self.cnn(self.OUT_STEPS, self.num_features_predicted, model_name='cnn',
                         kernel=parameters['kernel'], layer_size=parameters['layer_size'])

            history_cnn = self.compile_and_fit(lr=parameters['lr'], CheckPoint=True,
                                                   epochs=MAX_EPOCHS,
                                                   patience=5)
        #plot_history(history_cnn)
        # multi_window.plot_renormalized(
        # model=model, train_mean=train_mean, train_std=train_std)
        if parameters['model'] == 'bidirectional_lstm':
         

            self.bidirectional_lstm(self.OUT_STEPS, self.num_features_predicted, model_name='bidirectional_lstm',
                                        layer_number=parameters['layer_number'], layer_size=parameters['layer_size'])

            history_birirectional_lstm = self.compile_and_fit(lr=parameters['lr'], CheckPoint=True, epochs=MAX_EPOCHS, patience=5)
            
        if parameters['model'] == 'linear_lstm':
         

            self.linear_lstm(self.OUT_STEPS, self.num_features_predicted, model_name='linear_lstm',
                                        layer_number=parameters['layer_number'], layer_size=parameters['layer_size'])

            history_birirectional_lstm = self.compile_and_fit(lr=parameters['lr'], CheckPoint=True, epochs=MAX_EPOCHS, patience=5)

            plot_history(history_birirectional_lstm)
            # multi_window.plot_renormalized(model=model, train_mean=train_mean, train_std=train_std)

        if parameters['model'] == 'multilinear':
            
            """
            Define and train MULTILINEAR model
            """
            self.multi_linear(self.OUT_STEPS, self.num_features_predicted, model_name='multilinear',
                                  layer_number=parameters['layer_number'], layer_size=parameters['layer_size'])
            history_multilinear = self.compile_and_fit(lr=parameters['lr'], CheckPoint=True,
                                                           epochs=MAX_EPOCHS,
                                                           patience=5)
           # plot_history(history_multilinear)
        if parameters['model'] == 'advanced_model':
            

            self.advanced_model(self.OUT_STEPS, self.num_features_predicted, model_name='advanced_model', conv_number=parameters['conv_number'],
                              layer_number=parameters['layer_number'], layer_size=parameters['layer_size'])                   
            history_advanced = self.compile_and_fit(lr=parameters['lr'], CheckPoint=True,
                                                           epochs=MAX_EPOCHS,
                                                           patience=5)
            
            plot_history(history_advanced)
            plt.savefig('pred.png')
        # multi_window.plot_renormalized(
        # model=model, train_mean=train_mean, train_std=train_std)

        if parameters['model'] == 'dense':
            
            """
            Define and train DENSE model
            """

            self.dense(self.OUT_STEPS, self.num_features_predicted, model_name='dense',
                           layer_number=parameters['layer_number'], layer_size=parameters['layer_size'])
            history_dense = self.compile_and_fit(lr=parameters['lr'], CheckPoint=True,
                                                     epochs=MAX_EPOCHS,
                                                     patience=5)
           # plot_history(history_dense)
        # multi_window.plot_renormalized(
        # model=model, train_mean=train_mean, train_std=train_std)
        if parameters['model'] == 'convlstm1d':
            
            """
            Define and train conv1dlstm model
            """

            self.convlstm1d(self.OUT_STEPS, self.num_features_predicted, model_name='convlstm1d',
                           layer_number=parameters['layer_number'], layer_size=parameters['layer_size'])
            history_dense = self.compile_and_fit(lr=parameters['lr'], CheckPoint=True,
                                                     epochs=MAX_EPOCHS,
                                                     patience=5)
            plot_history(history_dense)
        # multi_window.plot_renormalized(
        # model=model, train_mean=train_mean, train_std=train_std)

    def loss(self, parameters):
        self.optimize_hyperparams(MAX_EPOCHS=self.max_epochs, parameters=parameters)
        print('prova nodi', parameters['layer_size'])
        print('prova conv', parameters['conv_number'])
        print('prova layer', parameters['layer_number'])
        return {'loss': self.multi_performance[parameters['model']][0], 'status': STATUS_OK}
    
    def prediction(self, parameters):
        self.optimize_hyperparams(MAX_EPOCHS=self.max_epochs, parameters=parameters)
        inputs, labels = self.example
        predictions = self.model.predict(inputs)
        predictions[0].reshape(12)
        predictions[1].reshape(12)
        predictions[2].reshape(12)

        print('prova nodi', parameters['layer_size'])
        print('prova lr', parameters['lr'])
        print('prova layer', parameters['layer_number'])

        return {'loss': 
                ((np.absolute(np.array(predictions[0]-labels[0]))).mean())*25.33, 
                'status': STATUS_OK}

    def finding_best(self, trial, evals, parameters, example):
        self.example=example
        return fmin(fn=self.prediction, space=parameters, algo=tpe.suggest, max_evals=evals, trials=trial)

    def plot_opt(self, trial):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
        xs = [t['tid']+1 for t in trial.trials]
        ys = [(t['result']['loss']) for t in trial.trials]
        ax1.set_xlim(xs[0] - 1, xs[-1] + 1)
        ax1.scatter(xs, ys, s=20, c='lightblue', edgecolors='black')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        xs = [(t['misc']['vals']['layer_size']) for t in trial.trials]
        ys = [t['result']['loss'] for t in trial.trials]
        ax2.scatter((xs), ys, s=20, c='red', edgecolors='black')
        ax2.set_xlabel('nodes')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        xs = [t['misc']['vals']['lr'] for t in trial.trials]
        ys = [t['result']['loss'] for t in trial.trials]
        ax3.scatter(xs, ys, s=20, c='green', edgecolors='black')
        ax3.set_xlabel('learning_rate')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        count=0

        for t in range(10):
            if (float(ys[t])<0.7):
                count = count+1

        print('il modello risulta affidabile al ', (count/10.0)*100.0, '%')

        xs = [(t['misc']['vals']['layer_number']) for t in trial.trials]
        ys = [(t['result']['loss']) for t in trial.trials]
        ax4.scatter(xs, ys, s=20, c='orange', edgecolors='black')
        ax4.set_xlabel('layer_number')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        plt.show()
        count=0



class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):

        return inputs[:, -12:, 0]