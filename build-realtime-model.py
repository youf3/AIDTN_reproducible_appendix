# -*- coding: utf-8 -*-
import os, shutil, argparse
from math import sqrt
from numpy import concatenate
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import keras
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1), shift means data moves i rows
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (columns[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (columns[j], i)) for j in range(n_vars)]
    # put it all together, axis =1 means join df by rows
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def load_data(file_path):
    dataset = read_csv(file_path)
    dataset.dropna(axis=0, how='any', inplace=True)
    return dataset

def normalize_and_make_series(dataset, look_back):
    
    values = dataset.values
    values = values.astype('float64')
    # normalize features    
    features_predict = ['NVMe_total_util','CPU', 'Memory_used']
    y_values = dataset[features_predict].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_y = scaler.fit_transform(y_values)
    # frame as supervised learning
    column_num = dataset.columns.size
    column_names = dataset.columns.tolist()
    reframed = series_to_supervised(scaled, column_names, look_back, 1)
    # drop columns we don't want to predict, only remain cpu which we want to predict
    drop_column = []
    for i in range((look_back+1) * column_num-2, (look_back + 1) * column_num):
        drop_column.append(i)
    reframed.drop(reframed.columns[drop_column], axis=1, inplace=True)
    return reframed, scaler

def split_data(dataset, reframed, look_back):
    column_num = dataset.columns.size
    values = reframed.values
    # split into input and outputs, the last column is the value of time t and treat it as label, namely train_y/test_y    
    x = values[:, :-3]
    y = values[:, -3:]
    # split trainset and testset
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=False)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape(train_X.shape[0], look_back, column_num)
    test_X = test_X.reshape(test_X.shape[0], look_back, column_num)
    return train_X, train_y, test_X, test_y

def build_model(look_back, train_X):
    acti_func = 'relu'
    neurons = 128
    loss = 'mse'
    batch_size = 8
    optimizer = 'adam'
    model = Sequential()
    model.add(Bidirectional(LSTM(neurons,
                                 activation=acti_func,
                                 return_sequences=True), input_shape=(look_back, train_X.shape[2])))
    model.add(Bidirectional(LSTM(neurons, activation=acti_func)))
    model.add(Dense(3))
    model.add(Activation('linear'))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def train_model(cluster):
    path_data_read = 'dataset/training_data/{}'.format(cluster)
    path_old_model = 'model/{}/old_model/'.format(cluster)
    path_train_model = 'model/{}/increase_model/'.format(cluster)
    files = os.listdir(path_data_read)
    print('Staring Incremental Training............')
    for file in files:
        file_path = path_data_read + '/' + file
        name = file_path.split('/')
        num = name[-1].split('.')
        testdata_path = 'dataset/train_validation_data/{}/{}'.format(cluster,num[0])
        modelsave_path = path_train_model + str(num[0]) + 'model.h5'

        if os.path.exists(path_train_model):
            models = os.listdir(path_train_model)        
            src = path_train_model + '/' + models[0]
            dst = path_old_model + '/' + models[0]
            modelread_path = path_train_model + '/' + models[0]        
        else:
            models = []
            os.makedirs(path_train_model)
            os.makedirs(path_old_model)
        
        dataset = load_data(file_path)
        look_back = 8
        if cluster == 'prp':
            features = ['NVMe_total_util','CPU', 'Memory_used', 'Goodput', 'num_workers']
        else:
            features = ['NVMe_total_util','CPU', 'Memory_used', 'NVMe_from_transfer', 'num_workers']
        dataset = dataset[features]
        
        reframed, scaler = normalize_and_make_series(dataset, look_back)
        train_X, train_y, test_X, test_y = split_data(dataset, reframed, look_back)       
        batch_size = 8

        if models == []:
            model = build_model(look_back, train_X)
        else:
            # Incremental training requires loading the model first
            model = load_model(modelread_path)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=0)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        history = model.fit(train_X, train_y, epochs=15, batch_size=batch_size, validation_data=(test_X, test_y),
                            verbose=1, shuffle=False, callbacks=[TensorBoard(log_dir='log'),earlystopper])
        
        train_predict = model.predict(train_X, batch_size)
        test_predict = model.predict(test_X, batch_size)
        test_X = test_X.reshape((test_X.shape[0] * look_back, test_X.shape[2]))
        test_X = test_X[:test_y.shape[0], 1:]

        inv_y = np.c_[test_y]
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, :]
        
        inv_yhat = np.c_[test_predict]
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, :]        

        # calculate root mean squared error
        test_score = np.sqrt(mean_squared_error(inv_y, inv_yhat))        

        conl = ['NVMe_total_util','CPU', 'Memory_used']
        pred = pd.DataFrame(data=inv_yhat, columns=conl)
        pred.to_csv(testdata_path + 'inv_yhat.csv', index=None)

        pred = pd.DataFrame(data=inv_y, columns=conl)
        pred.to_csv(testdata_path + 'inv_y.csv', index=None)        

        model_json = model.to_json()

        if models != []:
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # move old model
            shutil.move(src, dst)
        model.save(modelsave_path)
    print('Training Finished. Model saved to {}'.format(modelsave_path))
    print('RMSE: %.2f ' % test_score)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', '-c', choices = ['prp','mrp'], default='mrp',  help='Cluster dataset to use. [prp, mrp]')    
    args = parser.parse_args()

    start_time = time.time()
    train_model(args.cluster)
    end_time = time.time()
    print('Training time : %s seconds' % ((end_time - start_time)))