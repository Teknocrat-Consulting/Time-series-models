import os,time
import pandas as pd
import matplotlib.pyplot as plt
from influxdb_client import InfluxDBClient, WriteOptions
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

class Write_Data_InfluxDB:
    def __init__(self,df,name,date_col):

        ## Read and load Data
        self.df = df
        self.name =  name # self.path.split("\\")[-1].split(".csv")[-2].strip()
        #print("Name : ",self.name)
        self.measurement = "Teknocrat_" + self.name
        #st.write(self.measurement)
        self.date_col = date_col


        self.date_formats = (
        '%b-%y','%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d', '%B %d, %Y', '%d-%b-%Y','%b %d, %Y', '%Y-%m',         
        '%m/%Y','%B %Y', '%Y-%m-%d %H:%M', '%d-%m-%Y %H:%M:%S','%m/%d/%Y %I:%M %p','%Y-%m-%dT%H:%M:%S''%a, %d %b %Y %H:%M:%S GMT', '%A, %B %d, %Y'   
        )

        ## Define env variables
        self.token = os.getenv('INFLUX_TOKEN')
        self.org = os.getenv('ORG')
        self.url = "http://localhost:8086"
        self.bucket = os.getenv("BUCKET")

    def parse_dates(self,date_str):
        for fmt in self.date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt, exact=False)
            except ValueError:
                continue  # Continue trying other formats
        return pd.NaT

        ## Writing Data to InfluxDB

    def write_data_to_influxdb(self,df):
        """
        This function reads a csv file and writes the data to Influx DB
        """
        try:
            with InfluxDBClient(url=self.url, token=self.token, org=self.org, debug = True,timeout=20000) as _client:

                with _client.write_api() as _write_client:

                    _write_client.write(self.bucket, self.org, record=df, data_frame_measurement_name=self.measurement)
        except Exception as e:
            print("Connection Error : ",e)

    def run(self):


        self.df[self.date_col] = self.df[self.date_col].apply(self.parse_dates)
        self.df.set_index(self.date_col,inplace=True)

        write_data = self.write_data_to_influxdb(self.df)
        print(f"Data Written to Influx DB in {self.measurement}!!")
        st.success(f"Data Written to Influx DB in {self.measurement}!!")

# path = r"C:\Users\rajpo\PycharmProjects\Tecnokrat\Time-series-models\Timeseries_dashboard\AAPL.csv"
# date_col = "timestamp"
# test  = Write_Data_InfluxDB(path,date_col)
# final_result = test.run()