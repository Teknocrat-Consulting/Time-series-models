# -*- coding: utf-8 -*-
"""Tutorial on using the InfluxDB client."""

import argparse

from influxdb import InfluxDBClient

import pandas as pd 
import numpy as np

def main(host, port=8086):
    """Instantiate a connection to the InfluxDB."""
    user = 'root'
    password = 'root'
    dbname = 'telegraf'
    dbuser = 'alex'
    dbuser_password = '123'
    partition = "'sda2'"
    time_zone = "tz('Asia/Kolkata')"

    query2 = 'select "device","used","host" from "disk" where "device" = {} order by time desc limit 10 {}'.format(partition, time_zone)

    # query_where = 'select Int_value from cpu_load_short where host=$host;'
    # bind_params = {'host': 'server01'}
    # json_body = [
    #     {
    #         "measurement": "cpu_load_short",
    #         "tags": {
    #             "host": "server01",
    #             "region": "us-west"
    #         },
    #         "time": "2009-11-10T23:00:00Z",
    #         "fields": {
    #             "Float_value": 0.64,
    #             "Int_value": 3,
    #             "String_value": "Text",
    #             "Bool_value": True
    #         }
    #     }
    # ]

    client = InfluxDBClient(host, port, user, password, dbname)

    # print("Create database: " + dbname)
    # client.create_database(dbname)

    # print("Create a retention policy")
    # client.create_retention_policy('awesome_policy', '3d', 3, default=True)

    # print("Switch user: " + dbuser)
    # client.switch_user(dbuser, dbuser_password)

    # print("Write points: {0}".format(json_body))
    # client.write_points(json_body)

   # print("Querying data: " + query1)
    
    print("Querying data: " + query2)
    result = client.query(query2)

    result_dict = list(result.get_points(measurement='disk', tags={'device': 'sda2','host':'AlexMLWS-20'}))
    
    df =pd.DataFrame()

    for i in result_dict:
        df = df.append(i, ignore_index=True)

    df.index = df["time"]

    df.index = pd.to_datetime(df.index)

   # df.index = df.index + pd.DateOffset(hours = 5, minutes = 30)

    df = df.drop(['time'],axis=1)

    print(df.head(df.shape[0]))


def parser(x):
    return pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

dfa=pd.read_csv('/home/stackup/csv-files/model-t4/train.csv',index_col=[0],date_parser=parser)

def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='example code to play with InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='10.0.0.147',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(host=args.host, port=args.port)
