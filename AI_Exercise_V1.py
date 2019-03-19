import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import warnings

warnings.filterwarnings('ignore')

# read data
raw = pd.read_csv('./data.csv', index_col='id')

def quick_clean(df, missing_value=0):
    """
    fill missing values with (missing_value), drop duplicated rows and sort by 'date'.
    """
    df = df.fillna(missing_value)
    df = df.drop_duplicates(list(df.columns),
                            keep=False)
    df = df.sort_values('date')
    return df

# data cleaning
raw = quick_clean(raw)


def df_to_ts_array(df):
    """
    (helper function of prepare_data())
    transform df to np.array with each row has:
        temperature
        precipitation
        y: temperature of the next day
    
    rtype: np.array
    """
    df_ = df.copy()
    df_['y'] = df_['temperature'].shift(periods=-1, fill_value=df_['temperature'].iloc[-1])
    df_ = df_.iloc[:-1]
    return np.array(df_[['temperature', 'precipitation', 'y']])


def split_sequences(sequences, n_steps):
    """
    (helper function of prepare_data())
    transform a time series np.array to X, y where
        X is of shape(number of samples, n_steps, 2)
        y is of shape(number of samples, )
    so we could input them into supervised model.
    
    rtypes: np.array
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def prepare_data(df, time_step=3, train_test_split_date='2018-07-31'):
    """
    transform df to X_train, y_train, X_test, y_test as np.arrays.
    One sample of X_train/X_test has shape(time_step, 2)
    
    train test split on a given date because shuffling time series would make
    some data in test set be seen during training.
    
    rtypes: np.array
    """
    
    train = df[df.date < train_test_split_date]
    test = df[df.date >= train_test_split_date]
    
    venues = list(df.venue.unique())
    
    X_train, y_train = np.empty((0, time_step, 2)), np.empty(0,)
    X_test,  y_test  = np.empty((0, time_step, 2)), np.empty(0,)

    for venue in venues:
        # train test
        train_venue = train[train.venue == venue]
        
        train_venue_arr = df_to_ts_array(train_venue)
            
        X_train_new , y_train_new = split_sequences(train_venue_arr, time_step)

        X_train = np.append(X_train, X_train_new, axis=0)
        y_train = np.append(y_train, y_train_new, axis=0)
        
        # then repeat for test set
        test_venue = test[test.venue == venue]
        
        test_venue_arr = df_to_ts_array(test_venue)
            
        X_test_new , y_test_new = split_sequences(test_venue_arr, time_step)
        
        X_test = np.append(X_test, X_test_new, axis=0)
        y_test = np.append(y_test, y_test_new, axis=0)
        
    print([a.shape for a in [X_train, y_train, X_test, y_test]])
    
    return X_train, y_train, X_test, y_test # np.array

X_train, y_train, X_test, y_test = prepare_data(raw, 5)

# design network
model = Sequential()
model.add(LSTM(1000,
               activation='relu',
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# fit network
reg = model.fit(X_train, 
                y_train, 
                epochs=50, 
                batch_size=X_train.shape[0], 
                validation_data=(X_test, y_test), 
                verbose=0, 
                shuffle=False)

# plot history
plt.plot(reg.history['loss'], label='train')
plt.plot(reg.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(X_test)

# calculate RMSE
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print('Test RMSE: %.3f' % rmse)

def temp_of_next_day(venue, time_step=3):
    """
    extract the last (time_step) days temperature and precipitation,
    then feed into model to predict next day's temperature.
    """
    X = np.array(raw[raw.venue == venue].iloc[-time_step:][['temperature', 'precipitation']]).reshape(1,time_step,2)
    y = int(model.predict(X))
    print(f"The next day's temperature at {venue} will be {y}°F.")
    return y

temp_of_next_day('Capital Center', 5)

# save model as model1.hdf5
model.save('./model1.hdf5')
