"""
InfluxDB Client Module

Handles all interactions with InfluxDB including connection management,
data reading, and writing operations.
"""

import argparse
from typing import Dict, List, Optional, Any
import pandas as pd
from influxdb import InfluxDBClient, DataFrameClient


class InfluxDBManager:
    """
    Manager class for InfluxDB operations.
    
    Attributes:
        host (str): InfluxDB host address
        port (int): InfluxDB port number
        username (str): Database username
        password (str): Database password
        database (str): Database name
        client (InfluxDBClient): InfluxDB client instance
        df_client (DataFrameClient): DataFrame client instance
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8086,
                 username: str = 'root', password: str = 'root',
                 database: str = 'telegraf'):
        """
        Initialize InfluxDB connection.
        
        Args:
            host: InfluxDB host address
            port: InfluxDB port number
            username: Database username
            password: Database password
            database: Database name
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        
        self.client = InfluxDBClient(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database
        )
        
        self.df_client = DataFrameClient(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database
        )
    
    def create_database(self, db_name: Optional[str] = None) -> None:
        """
        Create a new database.
        
        Args:
            db_name: Name of the database to create
        """
        if db_name is None:
            db_name = self.database
        self.client.create_database(db_name)
        print(f"Database '{db_name}' created successfully")
    
    def create_retention_policy(self, policy_name: str, duration: str,
                               replication: int = 1, default: bool = True) -> None:
        """
        Create a retention policy for the database.
        
        Args:
            policy_name: Name of the retention policy
            duration: Duration for data retention (e.g., '3d', '1w')
            replication: Replication factor
            default: Whether to set as default policy
        """
        self.client.create_retention_policy(
            policy_name, duration, replication, default=default
        )
        print(f"Retention policy '{policy_name}' created")
    
    def query_data(self, query: str, bind_params: Optional[Dict] = None) -> Any:
        """
        Execute a query on the database.
        
        Args:
            query: InfluxQL query string
            bind_params: Optional parameters to bind to the query
            
        Returns:
            Query result
        """
        result = self.client.query(query, bind_params=bind_params)
        return result
    
    def get_disk_usage_data(self, measurement: str = 'disk',
                           partition: str = 'sda2',
                           host: str = None,
                           limit: int = 100,
                           timezone: str = "Asia/Kolkata") -> pd.DataFrame:
        """
        Retrieve disk usage data from InfluxDB.
        
        Args:
            measurement: Measurement name
            partition: Disk partition to query
            host: Hostname to filter by
            limit: Number of records to retrieve
            timezone: Timezone for data
            
        Returns:
            DataFrame with disk usage data
        """
        tz_str = f"tz('{timezone}')" if timezone else ""
        
        query = f'''
            SELECT "device", "used", "host" 
            FROM "{measurement}" 
            WHERE "device" = '{partition}'
            {f'AND "host" = {host}' if host else ''}
            ORDER BY time DESC 
            LIMIT {limit} 
            {tz_str}
        '''
        
        result = self.query_data(query)
        
        # Convert to DataFrame
        points = list(result.get_points(measurement=measurement))
        if not points:
            return pd.DataFrame()
        
        df = pd.DataFrame(points)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        return df
    
    def write_dataframe(self, df: pd.DataFrame, measurement: str,
                       tags: Optional[Dict] = None, protocol: str = 'line') -> None:
        """
        Write a pandas DataFrame to InfluxDB.
        
        Args:
            df: DataFrame to write
            measurement: Measurement name
            tags: Optional tags to add
            protocol: Protocol to use ('line', 'json')
        """
        self.df_client.write_points(
            df, measurement, tags=tags, protocol=protocol
        )
        print(f"Data written to measurement '{measurement}'")
    
    def write_points(self, points: List[Dict]) -> None:
        """
        Write data points to InfluxDB.
        
        Args:
            points: List of point dictionaries
        """
        self.client.write_points(points)
        print(f"Written {len(points)} points to database")
    
    def close(self) -> None:
        """Close the database connection."""
        self.client.close()
        print("Database connection closed")


class UDPInfluxClient:
    """
    UDP client for sending data to InfluxDB.
    
    Attributes:
        udp_port (int): UDP port number
        client (InfluxDBClient): InfluxDB client configured for UDP
    """
    
    def __init__(self, udp_port: int):
        """
        Initialize UDP InfluxDB client.
        
        Args:
            udp_port: UDP port number from influxdb.conf
        """
        self.udp_port = udp_port
        self.client = InfluxDBClient(use_udp=True, udp_port=udp_port)
    
    def send_batch(self, json_body: Dict) -> None:
        """
        Send batch data via UDP.
        
        Args:
            json_body: Data to send in the format:
                {
                    "tags": {...},
                    "time": "...",
                    "points": [...]
                }
        """
        self.client.send_packet(json_body)
        print(f"Batch sent via UDP port {self.udp_port}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='InfluxDB client for storage forecasting'
    )
    parser.add_argument(
        '--host', type=str, default='localhost',
        help='InfluxDB host address'
    )
    parser.add_argument(
        '--port', type=int, default=8086,
        help='InfluxDB port number'
    )
    parser.add_argument(
        '--username', type=str, default='root',
        help='Database username'
    )
    parser.add_argument(
        '--password', type=str, default='root',
        help='Database password'
    )
    parser.add_argument(
        '--database', type=str, default='telegraf',
        help='Database name'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Example usage
    db_manager = InfluxDBManager(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        database=args.database
    )
    
    # Get disk usage data
    df = db_manager.get_disk_usage_data(limit=10)
    print(df.head())
    
    db_manager.close()