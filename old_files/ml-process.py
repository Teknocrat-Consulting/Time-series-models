import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
import warnings
from influxdb import DataFrameClient
from influxdb import InfluxDBClient
import argparse

warnings.filterwarnings('ignore')

boxname='Anygraf4'

fpts = 15

conf_per =0.85

deviation_scale = 1.96

deviation_window = 10

slope_window = 15

result=pd.DataFrame()


def adftest(c):
    result=adfuller(c)
    if result[1]<=0.05:
        return True
    elif result[1]>0.05:
        return False
    else:
        return None

def datacompat(ar,k,cf):
    preds=[np.nan]*(k-1)
    preds=preds+ar[:cf+1]
    return preds


i=np.zeros(5)
i =i.tolist()


def linreg(b):
    n=len(b)
    s=0
    for i in range(n):
        s=s+np.power(i,2)
    denom=s-n*np.power(np.mean(range(0,n)),2)
    ns=0
    for i in range(n):
        ns=ns+i*float(b[i])
    
    numer=ns-n*np.mean(b)*np.mean(range(0,n))
    
    k=numer/denom
    
    return k

def normal_dist_exception(b,scale):

    if abs(b[-1]-np.mean(b)) > abs(np.std(b)*scale - np.mean(b)):
        return True

    else:
        return False

def main(host, port=8086):
    """Instantiate a connection to the InfluxDB."""
    user = 'root'
    password = 'root'
    dbname = 'Box1'
    dbuser = 'alex'
    dbuser_password = '123'
    measurement = 'disk'

    query2 = 'select * from {} order by desc limit 10'.format(measurement)

    client = InfluxDBClient(host, port, user, password, dbname)

    print("Querying data: " + query2)
    
    result = client.query(query2)

    result_dict = list(result.get_points(measurement='disk', tags={'hostname': 'Box1'}))
    
    df =pd.DataFrame()
    for i in result_dict:
        df = df.append(i, ignore_index=True)

    df.index = df["time"]

    df.index = pd.to_datetime(df.index)

    df = df.drop(['time'],axis=1)


    print(df.head())


def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='example code to play with InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='10.0.0.202',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(host=args.host, port=args.port)


