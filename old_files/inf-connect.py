import argparse
import os
import json
import time
import subprocess
import psutil
import pandas as pd
import datetime as dt
from influxdb import DataFrameClient
from influxdb import InfluxDBClient

hostname = str(subprocess.check_output('hostname', shell = True))[2:-3] + '_test'

def non_root_abstract(f):
    
    tot_disk_used = 0
    
    user_file_systems = ['ext','nfs','zfs','xfs']
    for i in f:
        if 'root' not in i.split()[0]:
            #print(i.split()[0])
            if any(j in i.split()[1] for j in user_file_systems): 
                tot_disk_used = tot_disk_used + psutil.disk_usage(i.split()[-1]).used
                print(i.split()[0])

    return tot_disk_used,'ext_disk'

def root_abstract(f):
    
    tot_disk_used = 0
    
    user_file_systems = ['ext','nfs','zfs','xfs']
    for i in f:
        if 'root' in i.split()[0]:
            #print(i.split()[0])
            if any(j in i.split()[1] for j in user_file_systems): 
                tot_disk_used = tot_disk_used + psutil.disk_usage(i.split()[-1]).used
                print(i.split()[0])

    return tot_disk_used,'primary_disk'

def main(host, port=8086):
    try:
        os.system('rm storestat')
    except:
        pass

    os.system('df -Th >storestat')
    
    f = open('storestat','r')
    
    tot_disk_used =  root_abstract(f)
    # tot_disk_used = non_root_abstract(f)


    f.close()
    #exit()

    temp_dict = {'time': dt.datetime.now(),'hostname':[hostname], 'usage': [tot_disk_used[0]]}

    df = pd.DataFrame.from_dict(temp_dict)

    df.index = df['time']
    del df['time']

    user = 'root'
    password = 'root'
    dbname = hostname
    protocol = 'line'

    retention_policy_duration = '1h' 
    shard_duration = '1m'

    #Instantiate the connection to the InfluxDB client.
    
    client = DataFrameClient(host, port, user, password, dbname)
    
    print('CREATE DATABASE "{}" WITH DURATION {} REPLICATION 1 SHARD DURATION {} NAME "r1"'.format(hostname, retention_policy_duration, shard_duration))
    client.query('CREATE DATABASE "{}" WITH DURATION {}	REPLICATION 1 SHARD DURATION {} NAME "r1"'.format(hostname, retention_policy_duration, shard_duration))
    
    client.switch_database(hostname)

    # json_body = [ 
    # {
    #     "measurement": "disk",
    #     "tags": {
    #         "hostname": hostname,
    #         },
    #     "time": dt.datetime.now().strftime("%d:%m:%Y:%H:%M:%S"),
    #     "fields": {
    #         "usage": tot_disk_used
    #     }
    # }
    # ]

    client.write_points(df,tot_disk_used[1],protocol='line')

    f.close()
    
    os.system('rm storestat')

    exit()

    
def parse_args():
    """Parse the args from main."""
    parser = argparse.ArgumentParser(
        description='example code to play with InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='localhost',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(host=args.host, port=args.port)