import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from datetime import timedelta
from LSTM_streamlit_module import LSTM_predictions_streamlit


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

    return model,X_test,indices_test[-1]

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
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded CSV file
    df = pd.read_csv(uploaded_file)
     # Determine the last date in the dataframe

    # Display uploaded data
    st.sidebar.write("Uploaded Data:")
    total_df = pd.concat([df.head(5),df.tail(5)],axis=0)
    st.sidebar.write(total_df)

    # Select target column and datetime column
    target_column = st.sidebar.selectbox("Select Target Column", df.columns)
    datetime_column = st.sidebar.selectbox("Select Datetime Column", df.columns)

    # Select number of epochs
    epochs = st.sidebar.slider("Select Number of Epochs", min_value=1, max_value=500, value=10)

    next_pred_days = st.sidebar.slider("Select Days for future Prediction", min_value=1, max_value=365, value=1)
    # Button to start predictions
    if st.sidebar.button("Start Predictions"):
        model1,test1,ind_date = make_predictions(df, target_column, datetime_column, epochs)
        pred_data = generate_future_data(model1,test1,next_pred_days,ind_date)
        centre_align(f"Future predictions for next {next_pred_days} Days")
        st.write(pred_data)
        centre_align(f"Future predictions for next {next_pred_days} Days")
        st.line_chart(pred_data)


