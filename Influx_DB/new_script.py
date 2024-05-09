import os,time
import pandas as pd
import matplotlib.pyplot as plt
from influxdb_client import InfluxDBClient, WriteOptions
from dotenv import load_dotenv
load_dotenv()


start_time = time.time()

class Connect_InfluxDB_df:
    def __init__(self,path,date_col,time_range,cols_to_keep="all"):

        ## Read and load Data
        self.path = path
        self.name = self.path.split("\\")[-1].split(".csv")[-2].strip()
        #print("Name : ",self.name)
        self.measurement = "ex_" + self.name
        self.date_col = date_col
        self.time_range = time_range
        self.cols_to_keep = cols_to_keep

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


    def fetch_data(self,col_select):
        client = InfluxDBClient(url="http://localhost:8086", token=self.token, org=self.org)

        query_api = client.query_api()
        time_col = ["_time"]
    
        columns_to_keep = time_col + col_select

        # Calculate the start time for the time range
        if self.time_range == 0:
            start_time = 0
        else:
            start_time = "|> range(start: -30d)" # f'-{self.time_range}y'

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
        
        # flux_query = f'''
        # from(bucket: "{self.bucket}")
        # |> range(start: 0)
        # |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
        # '''
        
        result_df = query_api.query_data_frame(flux_query)
        return result_df
    

    def run(self):
        
        df = pd.read_csv(self.path)
        df[self.date_col] = df[self.date_col].apply(self.parse_dates)
        df.set_index(self.date_col,inplace=True)

        write_data = self.write_data_to_influxdb(df)
        print("Data Written to Influx DB!!")

        if self.cols_to_keep == "all":
            col_select = list(df.columns)
        else:
            col_select = self.cols_to_keep

        print("Columns choosen : ", col_select)
        result_df = self.fetch_data(col_select)
        print("Data Fetched from Influx DB!!")

        result_df['_time'] = result_df['_time'].dt.tz_localize(None)

        result_df.rename(columns={"_time":"timestamp"},inplace=True)

        result_df.to_csv(f"{self.name}" + "_ex.csv")
        print("Saved as : ",f"{self.name}" + "_ex.csv")

        return result_df

# path = r"C:\Users\rajpo\PycharmProjects\Tecnokrat\Time-series-models\Timeseries_dashboard\AAPL.csv"
# date_col = "timestamp"
# time_range = 0
# cols_to_keep = ["Open","High","Low"]
# test  = Connect_InfluxDB_df(path,date_col,time_range,cols_to_keep)
# final_result = test.run()

# df_orig = pd.read_csv(path)
# if len(df_orig) == len(final_result):
#     print("Success!!")
# else:
#     print("Failure")

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")

#print("final_result : ",final_result.info)
#print(final_result)