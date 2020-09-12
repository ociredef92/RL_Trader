import gzip
import json
import glob, os

from datetime import datetime, timedelta, time

import pandas as pd
import numpy as np

class LOBData:

    # initialize class attributes: root_path is the root folder, security is the currency pair to unpack
    def __init__(self, raw_data_path, security, root_caching_folder, frequency = timedelta(seconds=10)):

        assert frequency >= timedelta(seconds=10), 'Frequency must be greater than 10 seconds'

        self.raw_data_path = raw_data_path
        self.security = security
        self.frequency = frequency
        self.caching_folder = f'{root_caching_folder}/{security}/{int(frequency.total_seconds())}'
        self.raw_resampled_folder = f'{self.caching_folder}/raw-resampled'

        os.makedirs(self.raw_resampled_folder, exist_ok=True)       

    # method that unzips a single file and returns the dictionary like content
    def load_json(self, date, time):

        path_string = f'{self.raw_data_path}/{self.security}/{date}/{date.replace("/", "")}_{time}.json.gz'

        with gzip.open(path_string, 'r') as f:  # gzip
            json_bytes = f.read()               # bytes

        json_str = json_bytes.decode('utf-8')   # decode bytes
        return json.loads(json_str)             # to python object

        # except IOError as e:
        #     print(e.errno)
        #     print(e)

    def resample_raw_files(self, start_date, end_date):
        numdays = (end_date - start_date).days

        string_dates = [datetime.strftime(start_date + timedelta(days=x), '%Y/%m/%d') for x in range(numdays)]
        string_hours = [str(i) if len(str(i))==2 else '0'+str(i) for i in range(24)]

        for date in string_dates: #test on a small portion of files
            for time in string_hours:
                raw_json = self.load_json(date, time)
                #print(f'{counter}, {data_processing.file_path(date, time)}')
                # TODO - save data with required frequency

    # method that unravels the nested json structure into a more manageable list of lists
    def unravel_json(self, date, time):
        list_quotes = []
        json_file = self.load_json(date, time)
        for key in json_file.keys():
            list_quotes.append(list(zip(
                                    [i[0] for i in json_file.get(key).get('asks')], # ask px
                                    [i[1] for i in json_file.get(key).get('asks')], # ask size
                                    [i[0] for i in json_file.get(key).get('bids')], # bid px
                                    [i[1] for i in json_file.get(key).get('bids')], # bid size
                                    list(range(len(json_file.get(key).get('bids')))), # ob level - assuming same for both
                                    [datetime.strptime(key[-15:], '%Y%m%d_%H%M%S')]*100  #datetime                                    
                                ))) 
        return list_quotes

root_path = '/home/pawel/Documents/LOB-data/new' # path where zipped files are stored
root_caching_folder = '/home/pawel/Documents/LOB-data/cache' # processed cached data folder
security = 'USDT_BTC'
frequency = timedelta(minutes=10)
start = datetime(2020, 4, 4) # first day we captured data
end = datetime(2020, 8, 20) # first day we captured data

# instantiate class
data = LOBData(root_path, security, root_caching_folder, frequency)
data.resample_raw_files(start, end)
