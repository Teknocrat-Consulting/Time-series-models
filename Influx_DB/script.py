import influxdb_client, os, time,csv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

start_time = time.time()

class Connect_InfluxDB:
    def __init__(self,path):

        ## Read and load Data
        self.path = path
        self.name = self.path.split("\\")[-1].split(".csv")[-2].strip()
        print("Name : ",self.name)
        self.measurement = self.name

        ## Define env variables
        token = os.getenv('INFLUX_TOKEN')
        self.org = os.getenv('ORG')
        url = "http://localhost:8086"
        self.bucket = os.getenv("BUCKET")

        ## Writing Data to InfluxDB

        self.client = influxdb_client.InfluxDBClient(url=url, token=token, org=self.org)


    def write_data_to_influxdb(self,csv_file,bucket,measurement,org):
        """
        This function reads a csv file and writes the data to Influx DB by each row
        
        """
        try:
            check = []
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # Get header row
                for row in reader:
                    data = [
                        {
                            "measurement": measurement,
                            "tags": {},
                            "fields": {headers[i]: row[i] for i in range(len(headers))},
                        }
                    ]

                    check.append(data)
                    # Write data to InfluxDB
                    write_api = self.client.write_api(write_options=SYNCHRONOUS)
                    write_api.write(bucket=bucket, org=org, record=data)

            return check
        except Exception as e:
            print("Connection Error : ",e)


    def fetch_and_clean_data(self,result):
        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        if data:
            df_from_query = pd.DataFrame(data)
            #df.to_csv('trial1_data.csv', index=False)
        else:
            print("No data returned from the query.")

        ## Process dataframe     
        df_from_query = df_from_query[["_value","_field"]]

        all_df = []
        for values in list(pd.unique(df_from_query._field)):
            #print(values)
            group = df_from_query.groupby(by="_field").get_group(values)
            date_df = df_from_query[df_from_query['_field'] == values]

            date_df = date_df.rename(columns={'_value': values}).drop('_field', axis=1)
            # Reset index
            date_df = date_df.reset_index(drop=True)
            all_df.append(date_df)

        combined_df = pd.concat(all_df,axis=1)
        print("Before dropping duplicates : " ,len(combined_df))
        combined_df.drop_duplicates(inplace=True)
        print("After dropping duplicates : " ,len(combined_df))
        print("Data processed and returned!!")
        return combined_df

    def run(self):

        ## Call function to write data to DB
        write_data = self.write_data_to_influxdb(self.path,self.bucket,self.measurement,self.org)

        ## Check if all data is written to the DB
        df_orig = pd.read_csv(self.path)
        if len(df_orig) == len(write_data):
            print("All data from CSV written to InfluxDB")

        ## Query the data (We get whole data from the bucket)
        query_api  = self.client.query_api()

        flux_query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: 0)
        |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
        '''

        result = query_api.query(flux_query)

        final_result = self.fetch_and_clean_data(result)
        print(final_result.info())
        print(final_result)
        final_result.to_csv(f"{self.name}" + "_ex.csv")
        return final_result

path = "daily-minimum-temperatures-in-me.csv"
test  = Connect_InfluxDB(path)
test.run()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")