#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import json
import glob, os

from datetime import datetime, timedelta, time

import pandas as pd
import numpy as np


# In[2]:


class Preprocessing:

    # initialize class attributes: root_path is the root folder, security is the currency pair to unpack
    def __init__(self, root_path, security, root_caching_folder):
        self.root_path = root_path
        self.security = security
        self.root_caching_folder = root_caching_folder

    # method that generates file path
    def file_path(self, date, time):
        path_string = f'{self.root_path}/{self.security}/{date}/{date.replace("/", "")}_{time}.json.gz'
        return path_string
    

    # method that unzips a single file and returns the dictionary like content
    def load_json(self, date, time):

        # Read zipped files
        with gzip.GzipFile(self.file_path(date, time), 'r') as fin:    # gzip
            json_bytes = fin.read()                        # bytes

        json_str = json_bytes.decode('utf-8')            # decode bytes
        return json.loads(json_str)                      # to python object


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
                                    [json_file.get(key).get('seq')]*100, # seq
                                    [json_file.get(key).get('isFrozen')]*100, # frozen flag
                                    [datetime.strptime(key[-15:], '%Y%m%d_%H%M%S')]*100  #datetime 
                                ) ) ) 
        return list_quotes


    # method that converts list of lists into a df
    def get_data_df(self, date, time):
        global df
        list_quotes = self.unravel_json(date, time)

        df =  pd.DataFrame([y for x in list_quotes for y in x], #flatten the list of lists structure
                          columns = ['Ask_Price', 'Ask_Size', 'Bid_Price', 'Bid_Size',
                                        'Level', 'Seq', 'isFrozen', 'Datetime']).sort_values(by = ['Datetime', 'Level'])

        df['Ask_Price'] = pd.to_numeric(df['Ask_Price'])
        df['Bid_Price'] = pd.to_numeric(df['Bid_Price'])
        
        df['Bid_Notional'] = df['Bid_Size'] * df['Bid_Price']
        df['Ask_Notional'] = df['Ask_Size'] * df['Ask_Price']

        return df


    def get_bbo(self, df = None, **kwargs):

        # For this to work you need either a df or date and time to be specified
        if df is None and 'date' in kwargs and 'time' in kwargs:
            df = self.get_data_df(date, time)

        # First level orderbook (bbo - best bid offer)
        df_bbo = df[df.Level == 0].copy()

        # bbo spread and mid px
        df_bbo['Mid_Price'] = (df_bbo['Ask_Price'] + df_bbo['Bid_Price'])/2
        df_bbo['spread'] = (df_bbo['Ask_Price'] - df_bbo['Bid_Price'])/df_bbo['Mid_Price']

        return df_bbo
    
    def get_bbo_bars(self, date, time, df_bbo = None,agg_freq = '1H', caching=True):
        
        '''
        Method that returns a dataframe grouped by the preferred frequency.
        You can either pass your own df, or pass date and time and it would go through all steps
        from the zipped file. Default agg_freq is 1H, alternatives are '30s', '1min' etc.
        '''
        
        if df_bbo is None:
            df_bbo = self.get_bbo(date=date, time=time)
        
        # bbo hourly bars
        
        df_bbo_bars = df_bbo.groupby(pd.Grouper(key='Datetime', freq=agg_freq)).agg({'Mid_Price':['mean', 'max', 'min', 
                                                                                            'first','last', 'count',
                                                                                            'std'],
                                                                               'spread': 'mean' }  ) 

        # rename columns
        df_bbo_bars.columns = ['mid_mean' , 'mid_high', 'mid_low', 'mid_open', 'mid_close', 'mid_#_obs', 
                       'mid_std', 'mean_spread']
        
        if caching:
            df_bbo_bars.to_csv(f'{root_caching_folder}/{security}/bbo.csv', mode='a', header=False)
        
        return df_bbo_bars

    
    def get_depth_bars(self, date, time, df_bbo = None,agg_freq = '1H', spread_threshold_tight=0.0025, 
                           spread_threshold_medium=0.0050, spread_threshold_wide=0.0100,
                            caching=True):
        
        #df = self.get_data_df(date, time)
        #print(self.df)
        if df_bbo is None:
            df_bbo = self.get_bbo(date=date, time=time)
        
        bid_df = pd.merge(df[['Datetime', 'Bid_Price', 'Bid_Size', 'Bid_Notional']], 
                  df_bbo[['Mid_Price', 'spread','Datetime']], 
                  how = 'left', left_on = 'Datetime', right_on = 'Datetime')
        
        bid_df['Bid_Spread'] = (bid_df['Mid_Price'] - bid_df['Bid_Price']) / bid_df['Mid_Price'] # check the math
        
        # Filter by spread threshold, only keeping tightest part of the order book
        bid_tight_depth = bid_df[bid_df['Bid_Spread'] <= spread_threshold_tight].groupby('Datetime')['Bid_Notional'].sum()
        bid_medium_depth = bid_df[bid_df['Bid_Spread'] <= spread_threshold_medium].groupby('Datetime')['Bid_Notional'].sum()
        bid_wide_depth = bid_df[bid_df['Bid_Spread'] <= spread_threshold_wide].groupby('Datetime')['Bid_Notional'].sum()
        
        # spread to ask - merge with bbo mid px, calculate respective spreads
        ask_df = pd.merge(df[['Datetime', 'Ask_Price', 'Ask_Size', 'Ask_Notional']], 
                  df_bbo[['Mid_Price', 'spread','Datetime']], 
                  how = 'left', left_on = 'Datetime', right_on = 'Datetime')

        ask_df['Ask_Spread'] = np.abs((ask_df['Mid_Price'] - ask_df['Ask_Price']) / ask_df['Mid_Price']) # check the math
                       
        ask_tight_depth = ask_df[ask_df['Ask_Spread'] <= spread_threshold_tight].groupby('Datetime')['Ask_Notional'].sum()
        ask_medium_depth = ask_df[ask_df['Ask_Spread'] <= spread_threshold_medium].groupby('Datetime')['Ask_Notional'].sum()
        ask_wide_depth = ask_df[ask_df['Ask_Spread'] <= spread_threshold_wide].groupby('Datetime')['Ask_Notional'].sum()
        
        bid_ask_depth_df = pd.concat([bid_tight_depth, bid_medium_depth, bid_wide_depth,
                                     ask_tight_depth, ask_medium_depth, ask_wide_depth], axis=1)
        bid_ask_depth_df.columns = ['bid_tight_depth', 'bid_medium_depth', 'bid_wide_depth',
                              'ask_tight_depth', 'ask_medium_depth', 'ask_wide_depth']
        
        ba_depth_bars = bid_ask_depth_df.reset_index().groupby(pd.Grouper(key='Datetime', freq=agg_freq)).mean()
        
        if caching:

            ba_depth_bars.to_csv(f'{root_caching_folder}/{security}/depth.csv', mode='a', header=False)
            
        return ba_depth_bars
    
    
    def caching_checks(self):
    
        # Create main caching folder - if it does not exist
        try:
            os.makedirs(root_caching_folder)
            print(f'created {root_caching_folder} folder')
        except FileExistsError:
            # directory already exists
            pass
    
        # Create subfolder for security of interest - if it does not exist
        try:
            os.makedirs(f'{root_caching_folder}/{security}')
            print(f'created {root_caching_folder}/{security} subfolder')
        except FileExistsError:
            # directory already exists
            pass
        
        # If the file does not exist, create depth with headers
        if os.path.exists(f'{root_caching_folder}/{security}/depth.csv'):
            self.cached_date_ranges('depth')
        else:
            pd.DataFrame([],columns=['bid_tight_depth','bid_medium_depth', 'bid_wide_depth', 'ask_tight_depth', 
                                    'ask_medium_depth', 'ask_wide_depth']
                                    ).to_csv(f'{root_caching_folder}/{security}/depth.csv', mode='w', header=True)
            print(f'Created {root_caching_folder}/{security}/depth.csv')

        # If the file does not exist, create bbo with headers
        if os.path.exists(f'{root_caching_folder}/{security}/bbo.csv'):
            #check daterange of data already cached
            self.cached_date_ranges('bbo')
        else:
            pd.DataFrame([], columns=['mid_mean' , 'mid_high', 'mid_low', 'mid_open', 'mid_close', 'mid_#_obs', 
                                      'mid_std', 'mean_spread']
                                    ).to_csv(f'{root_caching_folder}/{security}/bbo.csv', mode='w', header=True)
            print(f'Created {root_caching_folder}/{security}/bbo.csv')


    
    def cached_date_ranges(self, df_type):
        if df_type == 'depth':
            date_range_depth = pd.read_csv(f'{root_caching_folder}/{security}/depth.csv', header=0, index_col=0).index
            #return date_range_depth
            print(f'''Depth data already cached:
                    Dataframe size: {date_range_depth.shape},
                    Start date range: {date_range_depth.min()}, 
                    End date range: {date_range_depth.max()},
                    {date_range_depth.duplicated().sum()} duplicated entries...
                    {date_range_depth[date_range_depth.duplicated()]}''')
        
        elif df_type == 'bbo':
            date_range_bbo = pd.read_csv(f'{root_caching_folder}/{security}/bbo.csv', header=0, index_col=0).index
            print(f'''Bbo data already cached:
                    Dataframe size: {date_range_bbo.shape},
                    Start date range: {date_range_bbo.min()}, 
                    End date range: {date_range_bbo.max()},
                    {date_range_bbo.duplicated().sum()} duplicated entries...
                    
                    {date_range_bbo[date_range_bbo.duplicated()]}''')
                


# In[3]:


root_path = '/Users/federicotampieri/Downloads' # path where zipped files are stored

root_caching_folder = '/Users/federicotampieri/Downloads/RL_Trader_Caching' # processed cached data folder
#security_root_caching_folder = f'{root_caching_folder}/{security}' # individual security folder

security = 'USDT_BTC'

agg_freq = '10min'

base = datetime(2020, 4, 4) # first day we captured data
numdays = (datetime.today() - base).days # lastday we captured data - prev day

string_dates = [datetime.strftime(base + timedelta(days=x), '%Y/%m/%d') for x in range(numdays)]
string_hours = [str(i) if len(str(i))==2 else '0'+str(i) for i in range(24)]


# In[4]:


# instantiate class
data_processing = Preprocessing(root_path, security, root_caching_folder)

# Create relevant files and folders if missing
data_processing.caching_checks()


# In[ ]:


# write to csv
counter = 1
for date in string_dates: #test on a small portion of files
    for time in string_hours:
        
        print(f'{counter}, {data_processing.file_path(date, time)}')
        
        try:
            df_bbo = data_processing.get_bbo(date=date, time=time)

            if df_bbo.shape[0] > 0:

                data_processing.get_bbo_bars(df_bbo=df_bbo, agg_freq=agg_freq ,date=date, time=time)

                data_processing.get_depth_bars(date, time, df_bbo=df_bbo, agg_freq=agg_freq)

            else:
                print(f'EMPTY FILE!: {data_processing.file_path(date, time)}')
            counter+=1

        except IOError as e:
            print(e.errno)
            print(e)


# In[6]:


#read csv
depth_df = pd.read_csv(f'{root_caching_folder}/{security}/depth.csv', header=0, index_col=0)
bbo_df = pd.read_csv(f'{root_caching_folder}/{security}/bbo.csv', header=0, index_col=0)


# In[66]:


bbo_df.mid_mean.plot()


# In[8]:


depth_df


# In[ ]:


# when you pass this code to the NN,
# create cache folder with different level of aggregation, check if that exists or not
# create aws account
# visualization


# In[6]:


#pd.merge(data_processing.get_bbo_bars(date, time, agg_freq='5min'), 
#         data_processing.get_bid_depth_bars(date, time, agg_freq='5min'), 
#         how = 'inner', left_index = True, right_index = True)

