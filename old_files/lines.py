from influxdb import DataFrameClient
import pandas as pd
import time
from datetime import datetime
import argparse

# def lp(df,measurement,tag_key,field,value,datetime):
#     lines= [str(df[measurement][d]) + "," 
#             + str(tag_key) + "=" + str(df[tag_key][d]) 
#             + " " + str(df[field][d]) + "=" + str(df[value][d]) 
#             + " " + str(int(time.mktime(df[datetime][d].timetuple()))) + "000000000" for d in range(len(df))]
#     return lines

def main(host='10.0.0.164', port=8086):

	df =pd.read_csv('result.csv')
	#df.head(df.shape[0])

	df.set_index('Dates',inplace=True)

	df.index =  pd.to_datetime(df.index) 

	#df = df.iloc[:,1:]


#	lines = lp(df,"Actual","Predictions","Upper_bound","Lower_bound","Dates")

	user = 'root'
	password = 'root'
	dbname = 'storage'
	protocol = 'line'

	client = DataFrameClient(host, port, user, password, dbname)

	print("Create database: " + dbname)
	client.create_database(dbname)

	print("Write DataFrame")
	client.write_points(df, 'demo', protocol=protocol)

	print('points written')

def parse_args():
    """Parse the args from main."""
    parser = argparse.ArgumentParser(
        description='example code to play with InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='10.0.0.164',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	main(host=args.host, port=args.port)