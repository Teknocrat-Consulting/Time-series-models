import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class LSTM_predictions_streamlit():
    def __init__(self,df,pred_column,date_col = "timestamp",epoch=10):
        self.df = df
        self.df.dropna(inplace=True)
        self.epoch = epoch
        self.pred_column = pred_column
        self.date_col = date_col
        self.date_formats = (
        '%b-%y','%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d', '%B %d, %Y', '%d-%b-%Y','%b %d, %Y', '%Y-%m',         
        '%m/%Y','%B %Y', '%Y-%m-%d %H:%M', '%d-%m-%Y %H:%M:%S','%m/%d/%Y %I:%M %p','%Y-%m-%dT%H:%M:%S''%a, %d %b %Y %H:%M:%S GMT', '%A, %B %d, %Y'   
        )
        print("Columns: ",self.df.columns)
        print("Data shape: ",self.df.shape)

    def parse_dates(self,date_str):
        for fmt in self.date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt, exact=False)
            except ValueError:
                continue  # Continue trying other formats
        return pd.NaT
    
    def clean_numeric_value(self,value):
        # Remove commas and any non-numeric characters except dot, percent, and minus sign
        value=str(value)
        cleaned_value = re.sub(r'[^\d.]', '', value)
        return str(cleaned_value)

    def preprocess_data(self):
        #print("preprocessing called")
        self.df[self.pred_column] =  self.df[self.pred_column].apply(self.clean_numeric_value)
        self.df[self.pred_column] = pd.to_numeric(self.df[self.pred_column], errors='coerce').astype(float)
        self.df[self.date_col] =  self.df[self.date_col].apply(self.parse_dates)
        print(self.df.info())
        self.df = self.df[[self.date_col,self.pred_column]]
        print("*"*60)

        data = self.df.filter([self.pred_column])
        dataset = data.values

        # Get the number of rows to train the model on
        training_data_len = int(np.ceil( len(dataset) * .80 ))
        train_data = self.df.iloc[0:int(training_data_len), :]
        test_data = self.df.iloc[training_data_len - 60: , :]   

        print("Train size :",train_data.shape)
        print("Test size :",test_data.shape)

        self.train_data = train_data[[self.pred_column, self.date_col]].to_numpy()
        self.test_data = test_data[[self.pred_column, self.date_col]].to_numpy()

        return self.train_data,self.test_data
        print("*"*60)

    def plot(self):
        return self.df

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
        model.add(LSTM(128, return_sequences=True,activation='relu', input_shape= (X_train.shape[1], 1)))
        model.add(LSTM(64, activation='relu',return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_loss(self,history):
        loss_data = pd.DataFrame({
        'Epoch': range(1, len(history.history['loss']) + 1),
        'Train Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']})

        loss_data.set_index('Epoch', inplace=True)
        return loss_data
    
    def evaluate_model(self,y_train,y_train_pred,y_test,y_test_pred):
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
       
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        return train_rmse,test_rmse,train_mae,test_mae
    
    def df_for_plot(self,indices,y,y_pred):
        data = pd.DataFrame({
        'Date': pd.to_datetime(indices),
        'actual': y.flatten(),
        'predicted': y_pred.flatten()})

        # Set 'train_indices' column as index
        data.set_index('Date', inplace=True)
        return data