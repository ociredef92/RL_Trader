import gzip
import json
import glob, os

from datetime import datetime, timedelta, time

import pandas as pd
import numpy as np

class LOBData:

    # initialize class attributes: root_path is the root folder, security is the currency pair to unpack
    def __init__(self, raw_data_path, security, root_caching_folder, frequency = timedelta(seconds=10), levels=10):

        assert frequency >= timedelta(seconds=10), 'Frequency must be greater than 10 seconds'

        self.raw_data_path = raw_data_path
        self.security = security
        self.frequency = frequency
        self.caching_folder = f'{root_caching_folder}/{security}'
        self.cache_file = f'{self.caching_folder}/data-cache.csv'
        self.levels = levels

        os.makedirs(self.caching_folder, exist_ok=True)

    def get_LOB_data(self): # TODO consider returning date rage

        if os.path.isfile(self.cache_file):
            return pd.read_csv(self.cache_file)

        else:
            all_files = []
            for root, subdirs, files in os.walk(f'{self.raw_data_path}/{self.security}'):
                for filename in files:
                    all_files.append(filename) if filename.endswith('.json.gz') else None
            all_files.sort()

            first = all_files[0].split('.')[0] # get 20200403_13 from 20200403_13.json.gz
            last = all_files[-1].split('.')[0]
            start_date = datetime.strptime(first, '%Y%m%d_%H')
            end_date = datetime.strptime(last, '%Y%m%d_%H')

            print('Processing raw hourly snapshot files:')
            processed_data = []

            # Loop through all time steps between start and end date instead
            # of looping through all_files list to check for missing files
            while start_date <= end_date:
                subdir = datetime.strftime(start_date, '%Y/%m/%d')
                filename = datetime.strftime(start_date, '%Y%m%d_%H.json.gz')
                path_string = f'{self.raw_data_path}/{self.security}/{subdir}/{filename}'
                print(path_string)

                try:
                    # load raw data
                    with gzip.open(path_string, 'r') as f:  # gzip
                        json_bytes = f.read()               # bytes
                    raw_one_hour_data = json.loads(json_bytes.decode('utf-8'))

                    for key in raw_one_hour_data.keys():
                        # unravel the nested json structure into a more manageable list of lists
                        processed_data.append(list(zip(
                            [i[0] for i in raw_one_hour_data.get(key)['asks'][0:self.levels]], # ask px
                            [i[1] for i in raw_one_hour_data.get(key)['asks'][0:self.levels]], # ask size
                            [i[0] for i in raw_one_hour_data.get(key)['bids'][0:self.levels]], # bid px
                            [i[1] for i in raw_one_hour_data.get(key)['bids'][0:self.levels]], # bid size
                            list(range(self.levels)), # ob level - assuming same for both
                            [key[-15:]] * self.levels  # datetime part of the key
                        )))

                except IOError as e:
                    print(e.errno)
                    print(e)
                    # TODO Handle missing files

                start_date += timedelta(hours=1)
            
            # unravel nested structure and force data types
            df = pd.DataFrame([y for x in processed_data for y in x], #flatten the list of lists structure
                            columns = ['Ask_Price', 'Ask_Size', 'Bid_Price', 'Bid_Size','Level', 'Datetime'])

            df['Ask_Price'] = df['Ask_Price'].astype('float64')
            df['Ask_Size'] = df['Ask_Size'].astype('float64')
            df['Bid_Price'] = df['Bid_Price'].astype('float64')
            df['Bid_Size'] = df['Bid_Size'].astype('float64')
            df['Level'] = df['Level'].astype('int64')
            df['Datetime'] = df['Datetime'].astype('str')
            
            df.to_csv(self.cache_file)
            return df

# TODO add method which returns data with different frequency

root_path = '/home/pawel/Documents/LOB-data/new' # path where zipped files are stored
root_caching_folder = '/home/pawel/Documents/LOB-data/cache' # processed cached data folder
security = 'BTC_ETH'

# instantiate class
data = LOBData(root_path, security, root_caching_folder)
df = data.get_LOB_data()
print(df.shape)