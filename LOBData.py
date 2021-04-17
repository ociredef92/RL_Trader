import gzip
import json
import os

from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed

class LOBData:

    # initialize class attributes: root_path is the root folder, security is the currency pair to unpack
    def __init__(self, raw_data_path, security, root_caching_folder, frequency = timedelta(seconds=10), levels=10):

        assert frequency >= timedelta(seconds=1), 'Frequency must be equal to or greater than 1 second'

        self.raw_data_path = raw_data_path
        self.security = security
        self.frequency = frequency
        self.caching_folder = f'{root_caching_folder}/{security}'
        self.levels = levels

        all_files = []
        for root, subdirs, files in os.walk(f'{self.raw_data_path}/{self.security}'):
            for filename in files:
                all_files.append(filename) if filename.endswith('.json.gz') else None
        all_files.sort()

        first = all_files[0].split('.')[0].split('-')[0] # get 20200403_13 from 20200403_13.json.gz or 20200403_13-1-2.json.gz
        last = all_files[-1].split('.')[0].split('-')[0]

        # set start and end dates to previous day to avoid incomplete days when adding new data
        self.start_date = datetime.strptime(first, '%Y%m%d_%H').replace(hour=0)
        self.end_date = datetime.strptime(last, '%Y%m%d_%H').replace(hour=23) - timedelta(1)

        os.makedirs(f'{self.caching_folder}/{self.levels}_levels/{int(self.frequency.total_seconds())}s', exist_ok=True)
        os.makedirs(f'{self.caching_folder}/{self.levels}_levels/original_frequency', exist_ok=True)

    def get_data(self): # TODO consider returning date rage
        self.transform_and_resample()
        resampled_csv_files = f'{self.caching_folder}/{self.levels}_levels/{int(self.frequency.total_seconds())}s/*.csv.gz'
        return dd.read_csv(resampled_csv_files, compression='gzip')

    def transform_and_resample(self): # TODO consider returning date rage
        print(f'Checking for cached data from {self.start_date} to {self.end_date}')

        # Loop through day directories
        date_to_process = self.start_date
        while date_to_process <= self.end_date:
            day_folder = datetime.strftime(date_to_process, '%Y/%m/%d')
            day_cache_file_name = f'{datetime.strftime(date_to_process, "%Y-%m-%d")}.csv.gz'
            freq = f'{int(self.frequency.total_seconds())}s'
            resampled_file_name = f'{self.caching_folder}/{self.levels}_levels/{freq}/{day_cache_file_name}'
            if os.path.isfile(resampled_file_name):
                print(f'Found {resampled_file_name}')
            else:
                print(f'Generating {resampled_file_name}')
                original_file_name = f'{self.caching_folder}/{self.levels}_levels/original_frequency/{day_cache_file_name}'
                if os.path.isfile(original_file_name):
                    day_data = pd.read_csv(original_file_name, parse_dates=['Datetime'])
                else:
                    # empty json and nested list every new day processed
                    raw_data = {} # empty dict to update with incoming json
                    processed_data = []

                    # Load all files in to a dictionary
                    for filename in os.listdir(f'{self.raw_data_path}/{self.security}/{day_folder}'):
                        #print(f'Reading {self.security}/{filename}')
                        raw_data_temp = self.load_data_file(f'{self.raw_data_path}/{self.security}/{day_folder}/{filename}')
                        raw_data.update(raw_data_temp)

                    # number of seconds in a day / frequencey in seconds
                    snapshot_count_day = int(24 * 60 * 60 / self.frequency.total_seconds())
                    if len(raw_data) != snapshot_count_day:
                        diff = snapshot_count_day - len(raw_data)
                        if diff > 0:
                            print(f'{diff} gaps in {original_file_name}')
                        else:
                            print(f'{diff * -1} additional data points in {original_file_name}')

                    #del(raw_data['BTC_XRP-20200404_000000'])

                    #TODO fix sequence order

                    raw_data_frame = pd.DataFrame.from_dict(raw_data, orient='index')
                    raw_data_frame.reset_index(inplace=True)
                    raw_data_frame['index'] = raw_data_frame['index'].str[-15:]
                    raw_data_frame['index'] = pd.to_datetime(raw_data_frame['index'], format='%Y%m%d_%H%M%S')
                    raw_data_frame.set_index('index',drop=True,inplace=True)
                    raw_data_frame.sort_index(inplace=True)
                    idx_start = date_to_process
                    idx_end = date_to_process + timedelta(days=1) - timedelta(seconds=1)
                    idx = pd.date_range(idx_start, idx_end, freq=freq)
                    raw_data_frame = raw_data_frame.reindex(idx).ffill().fillna(method='bfill') # forward fill gaps and back fill first item if missing

                    # Convert hierarchical json data in to tabular format
                    levels = list(range(self.levels))
                    for row in raw_data_frame.itertuples():

                        ask_price, ask_volume = zip(* row.asks[0:self.levels])
                        bid_price, bid_volume = zip(* row.bids[0:self.levels])
                        sequences = [row.seq] * self.levels
                        datetimes = [row.Index] * self.levels

                        processed_data.append(list(zip(
                            ask_price,
                            ask_volume,
                            bid_price,
                            bid_volume,
                            levels,
                            sequences,
                            datetimes
                        )))

                    # unravel nested structure and force data types
                    day_data = pd.DataFrame([y for x in processed_data for y in x], #flatten the list of lists structure
                                    columns = ['Ask_Price', 'Ask_Size', 'Bid_Price', 'Bid_Size','Level', 'Sequence','Datetime'])

                    day_data['Ask_Price'] = day_data['Ask_Price'].astype('float64')
                    day_data['Bid_Price'] = day_data['Bid_Price'].astype('float64')
                    day_data['Sequence'] = day_data['Sequence'].astype('int64')

                    day_data.to_csv(original_file_name, compression='gzip')

                # resample dataframe to the wanted frequency
                resampled_day_data = day_data.groupby([pd.Grouper(key='Datetime', freq=freq), pd.Grouper(key='Level')]).last().reset_index()
                resampled_day_data.to_csv(resampled_file_name, compression='gzip')

            date_to_process += timedelta(days=1) # the most nested folder is a day of the month 

        return

    def load_data_file(self, path):
        try:
            with gzip.open(path, 'r') as f:
                json_string = f.read().decode('utf-8')
                frozen = json_string.count('"isFrozen": "1"')
                if frozen > 0:
                    print(f'Frozen {frozen} snapshots')
            return self.load_json(json_string)

        except Exception as e:
            print(e.errno)
            print(e)

    def load_json(self, json_string):
        try:
            json_object = json.loads(json_string)

        except json.JSONDecodeError as e:
            print(f'Malformed JSON in file at position {e.pos}')

            if '}0254}' in json_string:
                fixed_json_string = json_string.replace('}0254}', '}')
                return self.load_json(fixed_json_string)

            if e.msg == 'Expecting value':
                prev_snapshot_start = json_string.rindex('{', 0, e.pos)

                if prev_snapshot_start == 0:
                    # {"BTC_ETH-20201008_030000": ,"BTC_ETH-20201008_030010": {"asks
                    fixed_json_string = json_string[:1] + json_string[e.pos+1:]

                else:
                    # 5634},"BTC_ETH-20200903_095550": ,"BTC_ETH-20200903_095600": {"as
                    prev_snapshot_end = json_string.rindex('}', 0, e.pos) + 1
                    #prev_snapshot = json_string[prev_snapshot_start:prev_snapshot_end]
                    fixed_json_string = json_string[:prev_snapshot_end] + json_string[e.pos:]

            elif e.msg == 'Extra data':
                if json_string[e.pos-2:e.pos] == '}}':

                    print(json_string[e.pos-12:e.pos+12])

                    # 922}},"BTC_
                    #fixed_json_string = json_string[:e.pos-1] + json_string[e.pos:]
                    fixed_json_string = json_string.replace('}}', '}') + '}' # at the end should be }}
                    #print(fixed_json_string[e.pos-13112:e.pos+13112])

                else:
                    # en": "0", "seq": 945674867}, "seq": 945674845},"BTC_ETH-20
                    previous_comma = json_string.rindex(',', 0, e.pos)
                    fixed_json_string = json_string[:previous_comma] + json_string[e.pos:]

            else:
                # "seq": 933840511}": 933840515},"BTC_ET
                # "seq": 934014002}4001},"BTC_
                next_comma = json_string.index(',', e.pos)
                fixed_json_string = json_string[:e.pos] + json_string[next_comma:]
            return self.load_json(fixed_json_string)

        for key, value in list(json_object.items()):
            if not value['bids'] or not value['asks']:
                del json_object[key]

        return json_object

# TODO add method which returns data with different frequency

# root_path = '/home/pawel/Documents/LOB-data/new-format' # path where zipped files are stored
# root_caching_folder = '/home/pawel/Documents/LOB-data/cache-new-format' # processed cached data folder
# security = 'USDT_BTC'

# data = LOBData(root_path, security, root_caching_folder, timedelta(seconds=3), 10)
# df = data.get_data()
# print('DataFrame loaded')
# # computed = df.compute()
# # print(computed.shape)
# # print(computed.head())
# # print(computed.tail())


# # df = df.repartition(npartitions=1)

# start_date = datetime.strftime(data.start_date, '%Y_%m_%d')
# end_date = datetime.strftime(data.end_date, '%Y_%m_%d')

# output_file_name = f'{security}--10lev--3sec--{start_date}--{end_date}.csv.gz'
# df.to_csv(f'{root_caching_folder}/{security}/{output_file_name}', compression='gzip', single_file = True)
# print('Saved CSV')

#df.to_parquet(f'/tmp/10-seconds.parquet', compression='gzip', engine='pyarrow', write_index=False)
