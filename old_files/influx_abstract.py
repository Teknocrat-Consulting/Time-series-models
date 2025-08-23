import argparse
import pandas as pd

from influxdb import DataFrameClient

from influxdb import InfluxDBClient

def main(host, port=8086):
    """Instantiate the connection to the InfluxDB client."""
    user = 'root'
    password = 'root'
    dbname = 'storage'
    protocol = 'line'

    client = DataFrameClient(host, port, user, password, dbname)

    # print("Create pandas DataFrame")
    # df = pd.DataFrame(data=list(range(30)),
    #                   index=pd.date_range(start='2015-11-16',
    #                                       periods=30, freq='H'), columns=['0'])

    df = pd.read_csv('/home/stackup/db_automation/result.csv',index_col=[0])

    print(df.index)
    df.index=pd.to_datetime(df.index)


    print("Create database: " + dbname)
    client.create_database(dbname)

    print("Write DataFrame")
    client.write_points(df, 'demo', protocol=protocol)

    #print("Write DataFrame with Tags")
    #client.write_points(df, 'demo', protocol=protocol)

    print("Read DataFrame")
    print(client.query("select * from demo"))

    # print("Delete database: " + dbname)
    # client.drop_database(dbname)


def parse_args():
    """Parse the args from main."""
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