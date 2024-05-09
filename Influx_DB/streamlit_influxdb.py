import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from datetime import timedelta
import os
import pickle
from keras.models import load_model
from keras.models import save_model
from LSTM_streamlit_module import LSTM_predictions_streamlit
from influxdb_client import InfluxDBClient
from write_data import Write_Data_InfluxDB
from fetch_data import Fetch_Data_InfluxDB
from get_columns import Get_column_names


scaler = MinMaxScaler(feature_range=(-1, 1))
# Define Streamlit app
st.title("Time Series Forecasting Dashboard")

def centre_align(text):
    return st.write(f"<div style='text-align: center;'>{text}</div>", unsafe_allow_html=True)

# Function to preprocess data and make predictions
def make_predictions(df, pred_column, date_col, epochs):
    predictor = LSTM_predictions_streamlit(df, pred_column, date_col, epochs)

    # Preprocess data
    train_data,test_data = predictor.preprocess_data()

    plot_over_time = predictor.plot()
    centre_align(f'{pred_column.upper()} over Time')
    st.line_chart(plot_over_time.set_index(date_col))

    #scaler = MinMaxScaler(feature_range=(-1, 1)) # (0,1)
    train_data[:, 0] = scaler.fit_transform(train_data[:, 0].reshape(-1, 1)).flatten()
    test_data[:, 0] = scaler.transform(test_data[:, 0].reshape(-1, 1)).flatten() 

    # print(test_data)   
    # print(len(test_data))
    # print(type(test_data))

    X_train, y_train,indices_train  = predictor.create_sequences_lstm(train_data)
    X_test, y_test,indices_test = predictor.create_sequences_lstm(test_data)

    y_train = np.array([np.array([item]) for item in y_train])
    y_test = np.array([np.array([item]) for item in y_test])

    model = predictor.create_model(X_train)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(X_test))
    assert not np.any(np.isnan(y_train))
    assert not np.any(np.isnan(y_test))

    # print("X_test : ",(X_test[-1]))
    # print("y_test : ",(y_test[-1]))
    # print("last date : ",indices_test[-1])
    
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,verbose=1,validation_data=(X_test,y_test))

    loss_df = predictor.train_loss(history)
    centre_align("Train vs Validation loss per epoch")
    st.line_chart(loss_df)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)
    y_train = scaler.inverse_transform(y_train.reshape(-1,1))
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))

    #check_test = pd.DataFrame({"date" : indices_test,"actual":y_test.flatten(),"predictions":y_test_pred.flatten()})
    

    train_rmse,test_rmse,train_mae,test_mae = predictor.evaluate_model(y_train,y_train_pred,y_test,y_test_pred)

    st.success("Train Loss Evaluation")
    st.write('Train RMSE:', train_rmse)
    st.write('Train MAE:', train_mae)
    st.success("Test Loss Evaluation")
    st.write('Test RMSE:', test_rmse)
    st.write('Test MAE:', test_mae)

    train_predictions = predictor.df_for_plot(indices_train,y_train,y_train_pred)
    test_predictions = predictor.df_for_plot(indices_test,y_test,y_test_pred)

    centre_align("Predictions on Train set")
    st.line_chart(train_predictions)

    centre_align("Predictions on Test set")
    st.line_chart(test_predictions)

    centre_align("Actual and Predicted scores")
    st.write(test_predictions)

    return model,X_test,indices_test[-1],scaler,plot_over_time.set_index(date_col)

def generate_future_data(model,test,num_days,ind_test):
    # Retrieve the last 12 values from the test set
    last_12_values = test[-1]
    #print("Size of last_12_values:", last_12_values.shape)
    #print(last_12_values)

    future_dates = []
    future_predictions = []

    # Generate predictions for each future day
    for i in range(num_days):
        # Prepare input data for the next day
        X_future = last_12_values.reshape((1, 12, 1))  # Reshape to match LSTM input shape

        # Predict the next day's value
        future_pred_scaled = model.predict(X_future)
        future_pred = scaler.inverse_transform(future_pred_scaled)[0][0]

        # Update last_12_values with the new prediction
        last_12_values = np.append(last_12_values, future_pred_scaled)[-12:].reshape((1, 12, 1))

        # Append the prediction and corresponding date to the results
        future_date = ind_test + timedelta(days=i+1)
        future_dates.append(future_date)
        future_predictions.append(future_pred)

    # Create a DataFrame with the predicted values and corresponding dates
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Value': future_predictions})
    future_df.set_index('Date', inplace=True)
    return future_df

# Upload dataset

token = os.getenv('INFLUX_TOKEN')
org = os.getenv('ORG')
url = "http://localhost:8086"
bucket = os.getenv("BUCKET")

client = InfluxDBClient(url=url, token=token, org=org)

def show_measurements():
    flux_query = f'''
    import "influxdata/influxdb/schema"
    
    schema.measurements(bucket: "{bucket}")
'''

# Query measurements
    query_api = client.query_api()
    result = query_api.query(flux_query)

# Extract measurement names from the result
    measurement_names = []
    for table in result:
        for record in table.records:
            measurement_names.append(record.get_value())
    measurement_names = [x for x in measurement_names if "teknocrat" in x.lower()]

    print("Measurement names:", measurement_names)
    selected_measurement = st.sidebar.selectbox("Select Measurement", measurement_names)

    return selected_measurement
    # Print selected measurement (optional)
    #st.sidebar.write("Selected Measurement:", selected_measurement)

# Function to show time range input
def show_time_range_input():
    years = st.number_input("Enter Time Range (in years)", min_value=0, value=1)
    return years


option = st.sidebar.radio("Choose an option", ["Add Data", "Extract Data","Train and Predict"])

if 'final_result' not in st.session_state:
    st.session_state.final_result = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None


if option == "Add Data":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Write data to InfluxDB
        df = pd.read_csv(uploaded_file)

    # Display uploaded data
        st.sidebar.write("Uploaded Data:")
        total_df = pd.concat([df.head(5),df.tail(5)],axis=0)
        st.sidebar.write(total_df)
        datetime_column = st.selectbox("Select Datetime Column", df.columns)
        file_name = uploaded_file.name.split("\\")[-1].split(".csv")[-2].strip()
        #st.write(file_name)
        
        if st.button("Add Data to Influx DB"):
            test  = Write_Data_InfluxDB(df,file_name,datetime_column)
            final_result = test.run()
            #st.success("Data Added Successfully!")

        #st.sidebar.write("")

elif option == "Extract Data":
    # Fetch data from InfluxDB
    #measurements = Fetch_Data_InfluxDB.get_measurements()
    measurement = show_measurements()
    #st.sidebar.success(measurement)
    
    #st.text_input("Enter the measurement name to extrat the data from")
    time_range = show_time_range_input()
    select_all_columns = st.checkbox("Select All Columns")
    if select_all_columns:
        columns_to_give = "all"
    else:
        columns = Get_column_names(measurement).fetch_column_names()
        columns_to_give = st.multiselect("Select Columns below :", columns)
        print("col to give " ,columns_to_give)

    if st.button("Fetch Data from Influx DB"):
        test  = Fetch_Data_InfluxDB(measurement,time_range,columns_to_give)
        final_result,file_name = test.run()
        st.success("Data fetched successfully!!")
        st.write("Sample DataFrame : ")
        total_df = pd.concat([final_result.head(5),final_result.tail(5)],axis=0)
        st.write(total_df)
        st.session_state.final_result = final_result
        st.session_state.file_name = file_name

elif option == "Train and Predict":    
    if 'model1' not in st.session_state:
        st.session_state.model1 = None
    if 'test1' not in st.session_state:
        st.session_state.test1 = None
    if 'ind_date' not in st.session_state:
        st.session_state.ind_date = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    sample_df = pd.concat([st.session_state.final_result.head(5),st.session_state.final_result.tail(5)],axis=0)
    centre_align("Sample Dataframe")
    st.write(sample_df)

    model_name = "model" + "_" + st.session_state.file_name

    def save_data(model, test_set, last_date, scaler, df_pl,folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        model_file_path = os.path.join(folder_name, model_name + '.keras')
        save_model(model, model_file_path)
        np.save(os.path.join(folder_name, st.session_state.file_name + '_test_set.npy'), test_set)
        with open(os.path.join(folder_name, st.session_state.file_name + '_last_date.pkl'), 'wb') as f:
            pickle.dump(last_date, f)

        with open(os.path.join(folder_name, st.session_state.file_name + '_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        df_pl.to_csv(os.path.join(folder_name, st.session_state.file_name + '_plot.csv'), index=True)


    def load_data(model_name):
        model_file_path = os.path.join(st.session_state.file_name, model_name + '.keras')
        model = load_model(model_file_path)

        test_set = np.load(os.path.join(st.session_state.file_name, st.session_state.file_name + '_test_set.npy'))

        with open(os.path.join(st.session_state.file_name, st.session_state.file_name + '_last_date.pkl'), 'rb') as f:
            last_date = pickle.load(f)

        with open(os.path.join(st.session_state.file_name, st.session_state.file_name + '_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

        df_pl = pd.read_csv(os.path.join(st.session_state.file_name, st.session_state.file_name + '_plot.csv'))
        print(df_pl.info())
        #print(df.shape)

        return model, test_set, last_date, scaler, df_pl

        # Select target column and datetime column
    target_column = st.sidebar.selectbox("Select Target Column", st.session_state.final_result.columns)
    datetime_column = st.sidebar.selectbox("Select Datetime Column", st.session_state.final_result.columns)

        # Button to start predictions

    folder_path = os.path.join(os.getcwd(), st.session_state.file_name)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"The model '{st.session_state.file_name}' is present already.")
        st.success("The model is already trained and saved, click below to get Future Predictions")

        next_pred_days = st.slider("Select Days for future Prediction", min_value=1, max_value=365, value=1)

        if st.button("Get Future Predictions"):
            loaded_model, loaded_test_set, loaded_last_date,scaler,plot_over_time = load_data(model_name)
            date_col = plot_over_time.columns[0]
                #print(date_col)
            centre_align("Overall Trend of Data till last available date")
            st.line_chart(plot_over_time.set_index(date_col))
            pred_data = generate_future_data(loaded_model, loaded_test_set, next_pred_days, loaded_last_date)
            centre_align(f"Future predictions for next {next_pred_days} Days")
            st.write(pred_data)
            centre_align(f"Future predictions for next {next_pred_days} Days")
            st.line_chart(pred_data)
    else:
        print(f"The model '{st.session_state.file_name}' has to be trained and saved.")

    #        #Select number of epochs
        epochs = st.sidebar.slider("Select Number of Epochs", min_value=1, max_value=500, value=10)
        next_pred_days = st.sidebar.slider("Select Days for future Prediction", min_value=1, max_value=365, value=1)
        if st.sidebar.button("Start Predictions"):
            model1,test1,ind_date,scaler,df_time = make_predictions(st.session_state.final_result, target_column, datetime_column, epochs)
                # print("model : ",type(model1))
                # print("X_test : ",type(test1))
                # print("last date : ", type(ind_date))

            st.session_state.model1 = model1
            st.session_state.test1 = test1
            st.session_state.ind_date = ind_date
            st.session_state.scaler = scaler
            st.session_state.df_time = df_time
            pred_data = generate_future_data(model1,test1,next_pred_days,ind_date)
            centre_align(f"Future predictions for next {next_pred_days} Days")
            st.write(pred_data)
            centre_align(f"Future predictions for next {next_pred_days} Days")
            st.line_chart(pred_data)

    #             #print("model_1 : ", model1)

        if st.sidebar.button("Save Model for Future Predictions"):
                # print("model : ",type(st.session_state.model1))
                # print("X_test : ",type(st.session_state.test1))
                # print("last date : ", type(st.session_state.ind_date))
                # print("file_name : ",file_name)
                save_data(st.session_state.model1, st.session_state.test1, st.session_state.ind_date, st.session_state.scaler,st.session_state.df_time, st.session_state.file_name)
                st.sidebar.success("Data Saved!!")



