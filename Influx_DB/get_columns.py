from influxdb_client import InfluxDBClient
import streamlit as st,os
from dotenv import load_dotenv
load_dotenv()

class Get_column_names:
    def __init__(self,measurement):

        self.measurement = measurement
        ## Define env variables
        self.token = os.getenv('INFLUX_TOKEN')
        self.org = os.getenv('ORG')
        self.url = "http://localhost:8086"
        self.bucket = os.getenv("BUCKET")

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

## Change the range to close by time values

    def fetch_column_names(self):
        try:
            # Define connection parameters
            client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

            # Construct InfluxQL query to fetch field values from the _field column
            query = f'from(bucket: "{self.bucket}") |> range(start:-1y) |> filter(fn: (r) => r["_measurement"] == "{self.measurement}") |> distinct(column: "_field")'

            # Execute the query
            result = client.query_api().query(org=self.org, query=query)

            # Extract column names from the query result
            column_names = [record.get_value() for table in result for record in table.records]
            #print(column_names)
            return column_names

        except Exception as e:
            print("Error:", e)
            return None
    
# measurement_name = "Teknocrat_AAPL"
# column_names = Get_column_names(measurement_name)
# print("Columns names : ",column_names.fetch_column_names())
