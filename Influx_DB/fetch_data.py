import os,time
import pandas as pd
import matplotlib.pyplot as plt
from influxdb_client import InfluxDBClient, WriteOptions
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

class Fetch_Data_InfluxDB:
    def __init__(self,measurement,time_range,cols_to_keep="all"):

        self.measurement = measurement
        self.time_range = time_range
        self.cols_to_keep = cols_to_keep

        ## Define env variables
        self.token = os.getenv('INFLUX_TOKEN')
        self.org = os.getenv('ORG')
        self.url = "http://localhost:8086"
        self.bucket = os.getenv("BUCKET")

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

# Get column names

        # query = f'SHOW FIELD KEYS ON "{self.bucket}" FROM "{self.measurement}"'
        # result_cols = self.client.query_api().query(org=self.org, query=query)

        # # Extract column names from the result
        # column_names = [field.key for field in result_cols[0].field_keys]

        # print("Column names : ",column_names)

#     def show_measurements(self):
#         flux_query = f'''
#         import "influxdata/influxdb/schema"
        
#         schema.measurements(bucket: "{self.bucket}")
#     '''

#     # Query measurements
#         query_api = self.client.query_api()
#         result = query_api.query(flux_query)

# # Extract measurement names from the result
#         measurement_names = []
#         for table in result:
#             for record in table.records:
#                 measurement_names.append(record.get_value())
#         measurement_names = [x for x in measurement_names if "teknocrat" in x.lower()]

#         print("Measurement names:", measurement_names)
#         selected_measurement = st.sidebar.selectbox("Select Measurement", measurement_names)

#         # Print selected measurement (optional)
#         st.sidebar.write("Selected Measurement:", selected_measurement)

    def fetch_data(self,col_select):

        query_api = self.client.query_api()
        time_col = ["_time"]

         # Calculate the start time for the time range
        if self.time_range == 0:
            start_time = 0
        else:
            start_time = f'-{self.time_range}y' #

        if len(col_select) == 0:
            flux_query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {start_time})
            |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
            '''
        else:
            columns_to_keep = time_col + col_select

            # Convert the list to a formatted string
            columns_string = ', '.join([f'"{col}"' for col in columns_to_keep])

            # Construct the Flux query with the formatted columns string and start time
            flux_query = f'''
                    from(bucket: "{self.bucket}")
                    |> range(start: {start_time})
                    |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                    |> keep(columns: [{columns_string}])
                    '''
        
        result_df = query_api.query_data_frame(flux_query)
        return result_df
    
    def process_data(self,final_result):
        all_df = []
        count = 0

        if type(final_result)==list:
            for all_table in final_result:
                for table in list(pd.unique(all_table.table)):
                    if count == 0:
                        df1 = all_table.groupby(by="table").get_group(table)[["_time","_value","_field"]]
                    else:
                        df1 = all_table.groupby(by="table").get_group(table)[["_value","_field"]]
                    field_name = list(pd.unique(df1._field))
                    df1.rename(columns={"_value":field_name[0]},inplace=True)
                    df1.drop(["_field"],axis=1,inplace=True)
                    df1.reset_index(inplace=True)
                    df1.drop(['index'],axis=1,inplace=True)
                    all_df.append(df1)
                    count = count + 1
            
            final_df = pd.concat(all_df,axis=1)
            return final_df
        
        else:
            for table in list(pd.unique(final_result.table)):
                if count == 0:
                    df1 = final_result.groupby(by="table").get_group(table)[["_time","_value","c"]]
                else:
                    df1 = final_result.groupby(by="table").get_group(table)[["_value","_field"]]
                field_name = list(pd.unique(df1._field))
                df1.rename(columns={"_value":field_name[0]},inplace=True)
                df1.drop(["_field"],axis=1,inplace=True)
                df1.reset_index(inplace=True)
                df1.drop(['index'],axis=1,inplace=True)
                all_df.append(df1)
                count = count + 1

            final_df = pd.concat(all_df,axis=1)
            return final_df
        
    def run(self):  

        if self.cols_to_keep == "all":
            col_select = []
        else:
            col_select = self.cols_to_keep

        print("Columns choosen : ", col_select)
        result_df_1 = self.fetch_data(col_select)
        print(f"Data Fetched from {self.measurement}")

        if len(col_select)>1:
            result_df_1 = result_df_1.drop(["result","table"],axis=1)
            result_df_1['_time'] = result_df_1['_time'].dt.tz_localize(None)
            result_df_1.rename(columns={"_time":"timestamp"},inplace=True)
            result_df_1.to_csv(f"{self.measurement}" + "_ex.csv")
            print("Saved as : ",f"{self.measurement}" + "_ex.csv")
            return result_df_1,f"{self.measurement}"
        else:
            result_df = self.process_data(result_df_1)
            result_df['_time'] = result_df['_time'].dt.tz_localize(None)
            result_df.rename(columns={"_time":"timestamp"},inplace=True)
            result_df.to_csv(f"{self.measurement}" + "_ex.csv")
            print("Saved as : ",f"{self.measurement}" + "_ex.csv")

            return result_df,f"{self.measurement}"
    

# measurement  = "Teknocrat_Electric_Production"
# time_range = 0
# cols_to_keep = ["Open","High","Low"]
# test  = Fetch_Data_InfluxDB(measurement,time_range)
# final_result,str_ex = test.run()

# print(str_ex)