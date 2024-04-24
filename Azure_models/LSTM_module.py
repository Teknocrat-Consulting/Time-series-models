import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import psutil
import time

# Use TensorFlow's Keras, not standalone Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM , GRU, Dense, Dropout

import warnings
warnings.filterwarnings("ignore")


def measure_usage(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        cpu_times_pre = process.cpu_times()
        memory_use_pre = process.memory_info().rss  # rss = Resident Set Size

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        cpu_times_post = process.cpu_times()
        memory_use_post = process.memory_info().rss

        cpu_time_used = (cpu_times_post.user - cpu_times_pre.user) + (cpu_times_post.system - cpu_times_pre.system)
        memory_used = memory_use_post - memory_use_pre
        execution_time = end_time - start_time

        print(f"Function: {func.__name__}")
        print(f"CPU time used: {cpu_time_used:.2f} sec")
        print(f"Memory used: {memory_used / (1024**2):.2f} MB")  # Convert bytes to MB
        print(f"Execution time: {execution_time:.2f} sec")
        return result
    return wrapper

class LSTM_predictions():
    def __init__(self,df,pred_column,date_col = "timestamp",epoch=10):
        self.df = df
        self.df.dropna(inplace=True)
        self.epoch = epoch
        print("Columns: ",self.df.columns)
        print("Data shape: ",self.df.shape)
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        self.df = self.df[[date_col,pred_column]]
        print("*"*60)

        data = df.filter([pred_column])
        dataset = data.values

        # Get the number of rows to train the model on
        training_data_len = int(np.ceil( len(dataset) * .80 ))
        train_data = df.iloc[0:int(training_data_len), :]
        test_data = df.iloc[training_data_len - 60: , :]   

        print("Train size :",train_data.shape)
        print("Test size :",test_data.shape)

        self.train_data = train_data[[pred_column, date_col]].to_numpy()
        self.test_data = test_data[[pred_column, date_col]].to_numpy()

        print("*"*60)

    def plot(self,df,pred_column):
        self.df[pred_column].plot(title=f'{pred_column} over Time')

    def create_sequences_lstm(self,data, seq_length=12):
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
    

    def create_model(self,X_train):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    @measure_usage
    def run(self):

        train_data = self.train_data
        test_data = self.test_data

        scaler = MinMaxScaler(feature_range=(-1, 1)) # (0,1)
        train_data[:, 0] = scaler.fit_transform(train_data[:, 0].reshape(-1, 1)).flatten()
        test_data[:, 0] = scaler.transform(test_data[:, 0].reshape(-1, 1)).flatten()    

        X_train, y_train,indices_train  = self.create_sequences_lstm(train_data)
        X_test, y_test,indices_test = self.create_sequences_lstm(test_data)

        y_train = np.array([np.array([item]) for item in y_train])
        y_test = np.array([np.array([item]) for item in y_test])

        train_indices = pd.to_datetime(indices_train)
        test_indices = pd.to_datetime(indices_test)

        # To check the data shape for LSTM
        print("Train and Test Size after preprocessing: ")

        print("X_train shape: ",X_train.shape)
        print("y_train shape: ",y_train.shape)
        print("X_test shape: ",X_test.shape)
        print("y_test shape: ",y_test.shape)
        print("*"*60)

        model = self.create_model(X_train)
        print("New model running")

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_test))
        assert not np.any(np.isnan(y_train))
        assert not np.any(np.isnan(y_test))

        #print(X_train[0],y_train[0])

        history = model.fit(X_train, y_train, batch_size=32, epochs=self.epoch,verbose=1,validation_data=(X_test,y_test))
        print("Training details :",history)
        print("*"*60)

        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Progress During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        # Predicting and inverse transform to original scale
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)
        y_train = scaler.inverse_transform(y_train.reshape(-1,1))
        y_test = scaler.inverse_transform(y_test.reshape(-1,1))

        # Calculate performance metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

        print('Train MSE:', train_mse)
        print('Train RMSE:', train_rmse)
        print('Train MAE:', train_mae)
        print('Train MAPE:', train_mape)
        print("*"*60)

        print('Test MSE:', test_mse)
        print('Test RMSE:', test_rmse)
        print('Test MAE:', test_mae)
        print('Test MAPE:', test_mape)

        train_indices = pd.to_datetime(train_indices)
        test_indices = pd.to_datetime(test_indices)

        # Plot training data
        plt.figure(figsize=(15,7))
        plt.plot(train_indices, y_train.flatten(), label='Actual Train', color='blue')
        plt.plot(train_indices, y_train_pred.flatten(), label='Predicted Train', color='orange')
        plt.title('Train Prediction')
        plt.xlabel('Time')
        plt.ylabel('Average CPU Usage')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot testing data
        plt.figure(figsize=(15,7))
        plt.plot(test_indices, y_test.flatten(), label='Actual Test', color='blue')
        plt.plot(test_indices, y_test_pred.flatten(), label='Predicted Test', color='orange')
        plt.title('Test Prediction')
        plt.xlabel('Time')
        plt.ylabel('Average CPU Usage')
        plt.legend()
        plt.grid(True)
        plt.show()

df_azure = pd.read_csv("azure.csv")
azure = LSTM_predictions(df_azure,"avg cpu",epoch=100)
azure.plot(df_azure,"avg cpu")
azure.run()
