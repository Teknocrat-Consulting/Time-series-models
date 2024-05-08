
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
mlflow.tensorflow.autolog()


from tensorflow.keras.layers import LSTM, Dense,Dropout

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from kerastuner.tuners import RandomSearch


df = pd.read_csv('workstation_data.csv')
pred_column = 'energy'
date_col = 'timestamp'
date_formats = (
        '%b-%y','%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d', '%B %d, %Y', '%d-%b-%Y','%b %d, %Y', '%Y-%m',
        '%m/%Y','%B %Y', '%Y-%m-%d %H:%M', '%d-%m-%Y %H:%M:%S','%m/%d/%Y %I:%M %p','%Y-%m-%dT%H:%M:%S''%a, %d %b %Y %H:%M:%S GMT', '%A, %B %d, %Y'
        )

def parse_dates(date_str):
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt, exact=False)
            except ValueError:
                continue  # Continue trying other formats
        return pd.NaT

def clean_numeric_value(value):
        # Remove commas and any non-numeric characters except dot, percent, and minus sign
        value=str(value)
        cleaned_value = re.sub(r'[^\d.]', '', value)
        return str(cleaned_value)



df[pred_column] =  df[pred_column].apply(clean_numeric_value)

df[pred_column] = pd.to_numeric(df[pred_column], errors='coerce').astype(float)

df[date_col] =  df[date_col].apply(parse_dates)

df.info()

df = df[[date_col,pred_column]]
print("*"*60)

data = df.filter([pred_column])
dataset = data.values

dataset

# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .80 ))
train_data = df.iloc[0:int(training_data_len), :]
test_data = df.iloc[training_data_len - 60: , :]

train_data

test_data

print("Train size :",train_data.shape)
print("Test size :",test_data.shape)

train_data = train_data[[pred_column, date_col]].to_numpy()
test_data = test_data[[pred_column, date_col]].to_numpy()

scaler = MinMaxScaler(feature_range=(-1, 1)) # or any other range you prefer, (-1,1) is for (0,1) normalization

#scaler = MinMaxScaler(feature_range=(-1, 1)) # (0,1)
train_data[:, 0] = scaler.fit_transform(train_data[:, 0].reshape(-1, 1)).flatten()
test_data[:, 0] = scaler.transform(test_data[:, 0].reshape(-1, 1)).flatten()

print(test_data)
print(len(test_data))
print(type(test_data))

def create_sequences_lstm(data, seq_length=12):
        xs = []
        ys = []
        indices = []
        for i in range(len(data)-seq_length-1):
            x = data[i:(i+seq_length), 0:1]  # Select only the feature columns
            y = data[i+seq_length, 0]  # Select the target value
            xs.append(x)
            ys.append(y)
            indices.append(data[i+seq_length, 1])  # Capture the timestamp for plotting
        return np.array(xs), np.array(ys), np.array(indices)

X_train, y_train,indices_train  = create_sequences_lstm(train_data)
X_test, y_test,indices_test = create_sequences_lstm(test_data)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')



def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('lstm_units_1', min_value=32, max_value=256, step=32),
                   return_sequences=True,
                   activation='relu',
                   input_shape=(X_train.shape[1], 1)))


    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))


    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=32, max_value=256, step=32),
                   activation='relu',
                   return_sequences=False))


    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16),
                    activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Adjust as needed
    executions_per_trial=3,  # Adjust as needed
    directory='my_dir',
    project_name='keras_tuner_example'
)

tuner.search(X_train, y_train,
             epochs=100,
             validation_data=(X_test, y_test))

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_model.summary())
print(best_hyperparameters.values)

# history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))
{'lstm_units_1': 160, 'dropout_1': 0.4, 'lstm_units_2': 256, 'dropout_2': 0.2, 'dense_units': 128}

# history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))

# import matplotlib.pyplot as plt

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()



