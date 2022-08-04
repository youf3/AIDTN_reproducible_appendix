from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb

def get_train_throughput_rmse_mrp():
    train_data = pd.read_csv('dataset/training_data/mrp_xgboost.csv')
    features = ['NVMe_total_util', 'CPU', 'Memory_used', 'num_workers']
    output = ['Goodput']
    x_pred = train_data[features]
    y = train_data[output]
    booster1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                                colsample_bytree=1, max_depth=7)

    train_X, test_X, train_y, test_y = train_test_split(x_pred, y, test_size=0.2, random_state=0)
    booster1.fit(train_X, train_y)
    prediction = booster1.predict(test_X)

    rmse = mean_squared_error(test_y, prediction, squared=False)
    mae = mean_absolute_error(prediction, test_y)
    mse = mean_squared_error(prediction, test_y)
    print('MRP training RMSE:', rmse)
    print('MRP training MSE:', mse)
    print('MRP training MAE:', mae)


def get_train_throughput_rmse_prp():
    train_data = pd.read_csv('dataset/training_data/prp_xgboost.csv')
    features = ['NVMe_total_util', 'CPU', 'Memory_used', 'num_workers']
    output = ['Goodput']
    x_pred = train_data[features]
    y = train_data[output]
    booster1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                                colsample_bytree=1, max_depth=7)

    train_X, test_X, train_y, test_y = train_test_split(x_pred, y, test_size=0.2, random_state=0)
    booster1.fit(train_X, train_y)
    prediction = booster1.predict(test_X)

    rmse = mean_squared_error(test_y, prediction, squared=False)
    mae = mean_absolute_error(prediction, test_y)
    mse = mean_squared_error(prediction, test_y)
    print('PRP training RMSE:', rmse)
    print('PRP training MSE:', mse)
    print('PRP training MAE:', mae)

if __name__ == '__main__':
    get_train_throughput_rmse_mrp()
    get_train_throughput_rmse_prp()
