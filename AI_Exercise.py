import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import warnings
warnings.filterwarnings("ignore")

raw = pd.read_csv('./data.csv', index_col='id')

cc = raw[raw.venue == 'Capital Center'].sort_values('date')

def df_to_supervised(df, max_days_after=10):
    """
    transform the original time series to a supervised learning dataset.
        df: pd.DataFrame, a multivariate time series
        max_days_after: int, prevent a prediction on a long time after.
        
    returns: df_, a pd.DataFrame prepared for supervised learning.
    """
    df_ = df.sort_values('date').copy()
    
    df_['date'] = pd.to_datetime(df_['date'])
    df_['date_after'] = pd.to_datetime(df_['date'].shift(periods=-1, fill_value=df_['date'].iloc[-1]))
    df_['days_after'] = (df_['date_after'] - df_['date']).dt.days
    df_['month'] = df_['date_after'].dt.month
    df_['y'] = df_['temperature'].shift(periods=-1, fill_value=df_['temperature'].iloc[-1])
    
    df_ = df_[['month', 'temperature', 'precipitation', 'days_after', 'y']]
    df_ = df_[df_['days_after'] <= max_days_after]
    
    # Last record does not have label(future temperature), so it is removed
    df_ = df_.iloc[:-1]
    
    return df_

cc_ready = df_to_supervised(cc)

def scale_split(df, train_ratio=0.65):
    """
    conduct scaling and train test split on df.
        df: pd.DataFrame, ready for training.
        train_ratio: float
    
    returns:
        X_train: np.array, 
        y_train: np.array, 
        X_test: np.array, 
        y_test: np.array,
        X_scaler: StandardScaler() that stores info for X
        y_scaler: StandardScaler() that soters info for y
    """
    n_train = int(df.values.shape[0]*0.65)
    X = df.drop('y', axis=1).values
    y = df['y'].values.reshape(df.shape[0], 1)
    
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)
    
    X_train, y_train = X[:n_train, :], y[:n_train]
    X_test, y_test = X[n_train:, :], y[n_train:]
    
    return X_train, y_train, X_test, y_test, X_scaler, y_scaler

X_train, y_train, X_test, y_test, X_scaler, y_scaler = scale_split(cc_ready)
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

n_train = y_train.shape[0]
n_test = y_test.shape[0]

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# fit network
reg = model.fit(X_train, 
                y_train, 
                epochs=170, 
                batch_size=X_train.shape[0], 
                validation_data=(X_test, y_test), 
                verbose=0, 
                shuffle=False)

# make prediction
y_pred = model.predict(X_test)

# calculate RMSE
rmse = mean_squared_error(y_scaler.inverse_transform(y_test),
                          y_scaler.inverse_transform(y_pred)) ** 0.5
print('\nTest RMSE: %.3f' % rmse)

X_190124 = X_scaler.fit_transform(np.array([1, 27.8, 0, 3]).reshape(4,1)).reshape(1,1,4)

y_190124 = round(float(y_scaler.inverse_transform(model.predict(X_190124))), 2)
print(f'Model tells us the temperature of Capital Center on 2019-01-24 would be {y_190124}°F')

model.save('./model0.hdf5')


