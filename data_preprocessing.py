import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gzip
import json
import os
import shutil
import boto3
from os import listdir
from os.path import isfile, join

import dask.dataframe as dd

from configuration import config

def intraday_vol_ret(px_ts, span=100):
    '''
    Function to return rolling daily returns and exponentially weighted volatility

    Arguments:
    px_ts -- pandas series with a datetime index
    span -- integer, represents the ewm decay factor in terms of span: α=2/(span+1), for span≥1
    '''

    assert isinstance(px_ts.index, pd.DatetimeIndex), "px series must have a datetime index"
    
    df = px_ts.index.searchsorted(px_ts.index-pd.Timedelta(days=1)) # returns a scalar array of insertion point
    df = df[df>0] # remove inserion points of 0
    df = pd.Series(px_ts.index[df-1], index=px_ts.index[px_ts.shape[0]-df.shape[0]:]) # -1 to rebase to 0, index is a shifted version
    ret = px_ts.loc[df.index]/px_ts.loc[df.values].values-1 # "today" / yesterday - 1 -> 1d rolling returns
    vol = ret.ewm(span=span).std() # exponential weighted std. Specify decay in terms of span

    return ret, vol

# Higher level workflow function to keep notebooks tidy
# 1) import_px_data looks for standardized cached files in Experiments/cache (top ob train/test and depth train/test)
# 2) if no file is found it would import the CSV for the non std file from Experiments/input
# 3) the loaded file is passed to standardized_data_cache() which uses standardize() to actual perform standardization

def import_px_data(frequency, pair, date_start, date_end, lob_depth, norm_type, roll):
    '''
    Function that loads preprocessed data ready to be shaped/used for the model to train.
    Experiment folder is the path where data has been cached. The other parameters are part of the
    unique cached file nomenclature. If the file does not exist, it is generated frrom the input data
    in the "else" block

    Arguments:
    frequency --  timedelta, the minimum time granularity (e.g. timedelta(seconds=10))
    pair -- string, curency pair to return (e.g.'USDT_BTC')
    date_start -- string, timeseries start
    date_end -- string, timeseries end
    lob_depth -- integer, how many levels of the order book to be considered
    norm_type -- string, can assume values of 'z' or 'dyn' for z-score or dynamic z-score
    roll -- integer, function of the granularity provided
    '''

    configuration = config()

    resampled_data_folder = configuration['folders']['resampled_data']
    frequency_seconds = int(frequency.total_seconds())

    # Data import - needs to be adjusted importing from several files using Dask
    quotes_file_name = f'{pair}--{lob_depth}lev--{frequency_seconds}sec--{date_start}--{date_end}.csv.gz'

    standardized_train_file = f'{resampled_data_folder}/{pair}/TRAIN--{norm_type}-{roll}--{quotes_file_name}'
    standardized_test_file = f'{resampled_data_folder}/{pair}/TEST--{norm_type}-{roll}--{quotes_file_name}'

    top_ob_train_file = f'{resampled_data_folder}/{pair}/TRAIN_TOP--{quotes_file_name}'
    top_ob_test_file = f'{resampled_data_folder}/{pair}/TEST_TOP--{quotes_file_name}'

    # standardized test file contains both trades and quotes
    if os.path.isfile(standardized_test_file): # testing for one of cache files, assuming all were saved
        # Import cached standardized data
        print(f'Reading cached {standardized_train_file}')
        train_dyn_df = pd.read_csv(standardized_train_file, index_col=1)
        print(f'Reading cached {standardized_test_file}')
        test_dyn_df = pd.read_csv(standardized_test_file, index_col=1)

        print(f'Reading cached {top_ob_train_file}')
        top_ob_train = pd.read_csv(top_ob_train_file, index_col=[0,1])
        print(f'Reading cached {top_ob_test_file}')
        top_ob_test = pd.read_csv(top_ob_test_file, index_col=[0,1])

    else: # check separately for quotes and trades input files

        quotes_data_input = get_lob_data(pair, date_start, date_end, frequency, lob_depth)
        quotes_data_input['Datetime'] = dd.to_datetime(quotes_data_input['Datetime'])
        #assert lob_depth == quotes_data_input['Level'].max() + 1 # number of levels of order book - maybe add extra + 1 for trades

        trades_data_input = get_trade_data(pair, date_start, date_end, frequency)
        trades_data_input['Datetime'] = dd.to_datetime(trades_data_input['Datetime'])

        # once input files have been correctly read from the input folder, it's time to create a single standardized cache for trades and quotes

        # TODO - concatenate Dask dataframes
        quotes_data_input_pd = quotes_data_input.compute()
        trades_data_input_pd = trades_data_input.compute()

        data = pd.concat([trades_data_input_pd, quotes_data_input_pd]).sort_values(by=['Datetime', 'Level'])

        roll = roll + 1 # +1 from extra level trades(level -1)
        train_dyn_df, test_dyn_df, top_ob_train, top_ob_test = standardized_data_cache(data, roll, lob_depth, standardized_train_file, standardized_test_file, top_ob_train_file, top_ob_test_file)

    # reset indexes, cast datetime type and clean unwanted columns
    train_dyn_df = train_dyn_df.reset_index()
    train_dyn_df['Datetime'] = pd.to_datetime(train_dyn_df['Datetime'])
    train_dyn_df.drop('Unnamed: 0', axis=1, inplace=True)

    test_dyn_df = test_dyn_df.reset_index()
    test_dyn_df['Datetime'] = pd.to_datetime(test_dyn_df['Datetime'])
    test_dyn_df.drop('Unnamed: 0', axis=1, inplace=True)

    top_ob_train = top_ob_train.reset_index()
    top_ob_train['Datetime'] = pd.to_datetime(top_ob_train['Datetime'])
    top_ob_train.drop('Unnamed: 0.1', axis=1, inplace=True)

    top_ob_test = top_ob_test.reset_index()
    top_ob_test['Datetime'] = pd.to_datetime(top_ob_test['Datetime'])
    top_ob_test.drop('Unnamed: 0.1', axis=1, inplace=True)

    return train_dyn_df, test_dyn_df, top_ob_train, top_ob_test


def standardized_data_cache(data, roll, lob_depth, standardized_train_file, standardized_test_file, top_ob_train_file, top_ob_test_file):
    # Train test split
    train_test_split = int((data.shape[0] / lob_depth) * 0.7) # slice reference for train and test
    train_timestamps = data['Datetime'].unique()[:train_test_split]
    test_timestamps = data['Datetime'].unique()[train_test_split:]

    train_cached_data = data[data['Datetime'].isin(train_timestamps)].set_index(['Datetime', 'Level'])
    test_cached_data = data[data['Datetime'].isin(test_timestamps)].set_index(['Datetime', 'Level'])

    print(f'Train dataset shape: {train_cached_data.shape} - Test dataset shape: {test_cached_data.shape}')

    roll_shift = roll+1 # rolling period for dyn z score - + 1 from shift in standardize()

    # Training
    # custom rolling standardization for px and size separately
    train_dyn_prices = standardize(train_cached_data[['Ask_Price', 'Bid_Price']], lob_depth, 'dyn_z_score', roll)
    train_dyn_volumes = standardize(train_cached_data[['Ask_Size', 'Bid_Size']], lob_depth, 'dyn_z_score', roll)
    train_dyn_df = pd.concat([train_dyn_prices, train_dyn_volumes], axis=1).reset_index() # concat along row index #1
    print(f'Saving {standardized_train_file}')
    train_dyn_df.to_csv(standardized_train_file, compression='gzip') # save standardized data to csv 

    top_ob_train = train_cached_data[train_cached_data.index.get_level_values(1)==0][roll_shift:] #3
    top_ob_train['Mid_Price'] = (top_ob_train['Ask_Price'] + top_ob_train['Bid_Price']) / 2
    top_ob_train['Spread'] = (top_ob_train['Ask_Price'] - top_ob_train['Bid_Price']) / top_ob_train['Mid_Price']
    top_ob_train['merge_index'] = top_ob_train.reset_index().index.values # useful for merging later
    print(f'Saving {top_ob_train_file}')
    top_ob_train.to_csv(top_ob_train_file, compression='gzip') # save top level not standardized to csv

    # print(f'Saving {standardized_data_folder}/{pair}/TRAIN_top--{norm_type}-{roll}--{input_file_name}')
    # train_dyn_df[train_dyn_df['Level']==0].to_csv(f'{standardized_data_folder}/{pair}/TRAIN_TOP--{norm_type}-{roll}--{input_file_name}', compression='gzip') # save top level to csv 

    # Test
    # custom rolling standardization for px and size separately
    test_dyn_prices = standardize(test_cached_data[['Ask_Price', 'Bid_Price']], lob_depth, 'dyn_z_score', roll)
    test_dyn_volumes = standardize(test_cached_data[['Ask_Size', 'Bid_Size']], lob_depth, 'dyn_z_score', roll)
    test_dyn_df = pd.concat([test_dyn_prices, test_dyn_volumes], axis=1).reset_index() # concat along row index #2
    print(f'Saving {standardized_test_file}')
    test_dyn_df.to_csv(standardized_test_file, compression='gzip') # save standardized data to csv

    top_ob_test = test_cached_data[test_cached_data.index.get_level_values(1)==0][roll_shift:] #4
    top_ob_test['Mid_Price'] = (top_ob_test['Ask_Price'] + top_ob_test['Bid_Price']) / 2
    top_ob_test['Spread'] = (top_ob_test['Ask_Price'] - top_ob_test['Bid_Price']) / top_ob_test['Mid_Price']
    top_ob_test['merge_index'] = top_ob_test.reset_index().index.values # useful for merging later
    print(f'Saving {top_ob_test_file}')
    top_ob_test.to_csv(top_ob_test_file, compression='gzip') # # save top level not standardized to csv

    return train_dyn_df, test_dyn_df, top_ob_train, top_ob_test

# Model training - data preparation
def standardize(ts, ob_levels, norm_type='z_score', roll=0):
    '''
    Function to standardize (mean of zero and unit variance) timeseries

    Arguments:
    ts -- pandas series or df having timestamp and ob level as index to allow sorting (dynamic z score)
    ob_levels -- number of ob levels analyzed
    norm_type -- string, can assume values of 'z' or 'dyn' for z-score or dynamic z-score
    roll -- integer, rolling window for dyanamic normalization.

    Returns: pandas series
    '''
    
    if norm_type=='z_score':
        
        try:
            if ts.shape[1] > 1:
                ts_stacked = ts.stack()
        except:
            ts_stacked = ts
        
        return (ts-ts_stacked.mean()) / ts_stacked.std()
    
    # dynamic can't accomodate multi columns normalization yet
    elif norm_type=='dyn_z_score' and roll>0:

        ts_shape = ts.shape[1]

        if ts_shape > 1:
            ts_stacked = ts.stack()

            print(f'rolling window = {roll * ob_levels * ts_shape}, calculate as roll: {roll} * levels: {ob_levels} * shape[1]: {ts_shape}')

            ts_dyn_z = (ts_stacked - ts_stacked.rolling(roll * ob_levels * ts_shape).mean().shift((ob_levels * ts_shape) + 1) 
              ) / ts_stacked.rolling(roll * ob_levels * ts_shape).std(ddof=0).shift((ob_levels * ts_shape) + 1)
            
            norm_df = ts_dyn_z.reset_index().pivot_table(index=['Datetime', 'Level'], columns='level_2', values=0, dropna=True)
            print('done')
            #Q.put(norm_df)
            return norm_df
    else:
        print('Normalization not perfmed, please check your code')

def get_lob_data(pair, date_start, date_end, frequency = timedelta(seconds=10), lob_depth=10):
    '''
    Function to get limit orde book snapshots time series

    Arguments:
    pair -- string, curency pair to return (e.g.'USDT_BTC')
    date_start -- string, timeseries start
    date_end -- string, timeseries end
    frequency -- timedelta, the minimum time granularity (e.g. timedelta(seconds=10))
    lob_depth -- number of ob levels analyzed

    Returns: Dask data frame
    '''

    print(f'Checking for cached LOB data from {date_start} to {date_end}')

    #TODO assert if date_end is yesterday or earlier

    assert frequency >= timedelta(seconds=1), 'Frequency must be equal to or greater than 1 second'

    configuration = config()
    raw_data_folder = configuration['folders']['raw_lob_data']
    resampled_data_folder = configuration['folders']['resampled_data']

    date_start = datetime.strptime(date_start, '%Y-%m-%d')
    date_end = datetime.strptime(date_end, '%Y-%m-%d')
    freq = f'{int(frequency.total_seconds())}s'

    os.makedirs(f'{resampled_data_folder}/{pair}/{lob_depth}_levels/original_frequency', exist_ok=True)
    os.makedirs(f'{resampled_data_folder}/{pair}/{lob_depth}_levels/{freq}', exist_ok=True)

    data = []

    # Loop through day folders
    date_to_process = date_start
    while date_to_process <= date_end:
        day_folder = datetime.strftime(date_to_process, '%Y/%m/%d')
        day_cache_file_name = f'{datetime.strftime(date_to_process, "%Y-%m-%d")}.csv.gz'
        resampled_file_path = f'{resampled_data_folder}/{pair}/{lob_depth}_levels/{freq}/{day_cache_file_name}'
        if os.path.isfile(resampled_file_path):
            print(f'Found {resampled_file_path}')
        else:
            print(f'Generating {resampled_file_path}')
            original_file_name = f'{resampled_data_folder}/{pair}/{lob_depth}_levels/original_frequency/{day_cache_file_name}'
            if os.path.isfile(original_file_name):
                day_data = pd.read_csv(original_file_name, parse_dates=['Datetime'])
            else:
                # empty json and nested list every new day processed
                raw_data = {} # empty dict to update with incoming json
                processed_data = []

                if not os.path.isdir(f'{raw_data_folder}/{pair}/{day_folder}'):

                    s3_resource = boto3.resource('s3')

                    """
                    The calls to AWS STS AssumeRole must be signed with the access key ID and secret access key of an existing IAM user.
                    The credentials can be in environment variables or in a configuration file and will be discovered automatically by the boto3.client() function.
                    For more information, see the Python SDK documentation: http://boto3.readthedocs.io/en/latest/reference/services/sts.html#client
                    """
                    if configuration['other'].getboolean('cross_account_access'):
                        sts_client = boto3.client('sts')
                        response = sts_client.assume_role(RoleArn=configuration['other']['cross_account_access_role'], RoleSessionName="AssumeRoleSession")
                        s3_resource = boto3.resource(
                            's3',
                            aws_access_key_id = response['Credentials']['AccessKeyId'],
                            aws_secret_access_key = response['Credentials']['SecretAccessKey'],
                            aws_session_token = response['Credentials']['SessionToken'],
                        )

                    lob_data_bucket = s3_resource.Bucket(configuration['buckets']['lob_data'])

                    os.makedirs(f'{raw_data_folder}/tmp/{pair}/{day_folder}', exist_ok=True)
                    for obj in lob_data_bucket.objects.filter(Prefix=f'{pair}/{day_folder}'):
                        lob_data_bucket.download_file(obj.key, f'{raw_data_folder}/tmp/{obj.key}')
                        print(f'Downloaded {obj.key} from S3')
                    shutil.move(f'{raw_data_folder}/tmp/{pair}/{day_folder}', f'{raw_data_folder}/{pair}/{day_folder}')

                # Load all files in to a dictionary
                for file_name in os.listdir(f'{raw_data_folder}/{pair}/{day_folder}'):

                    try:
                        with gzip.open(f'{raw_data_folder}/{pair}/{day_folder}/{file_name}', 'r') as f:
                            json_string = f.read().decode('utf-8')
                            frozen = json_string.count('"isFrozen": "1"')
                            if frozen > 0:
                                print(f'Frozen {frozen} snapshots')
                        raw_data_temp = load_lob_json(json_string)

                    except Exception as e:
                        print(e.errno)
                        print(e)

                    raw_data.update(raw_data_temp)

                # number of seconds in a day / frequencey in seconds
                snapshot_count_day = int(24 * 60 * 60 / frequency.total_seconds())
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
                levels = list(range(lob_depth))
                for row in raw_data_frame.itertuples():

                    ask_price, ask_volume = zip(* row.asks[0:lob_depth])
                    bid_price, bid_volume = zip(* row.bids[0:lob_depth])
                    sequences = [row.seq] * lob_depth
                    datetimes = [row.Index] * lob_depth

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
            resampled_day_data.to_csv(resampled_file_path, compression='gzip')

        date_to_process += timedelta(days=1) # the most nested folder is a day of the month 
        data.append(resampled_file_path)

    # computed = df.compute()
    # df = df.repartition(npartitions=1)
    # df.to_csv(f'{root_caching_folder}/{pair}/{output_file_name}', compression='gzip', single_file = True)
    # df.to_parquet(f'/tmp/10-seconds.parquet', compression='gzip', engine='pyarrow', write_index=False)

    return dd.read_csv(data, compression='gzip')

def load_lob_json(json_string):
    '''
    Function decode json and fix malformed data issues.
    Calls itself recursvely on exceptions until all issues are fixed. 

    Arguments:
    json_string -- string, json to decode and fix

    Returns: dictionary from decoded json string
    '''
    try:
        json_dict = json.loads(json_string)

    except json.JSONDecodeError as e:
        print(f'Malformed JSON in file at position {e.pos}')

        if '}0254}' in json_string:
            fixed_json_string = json_string.replace('}0254}', '}')
            return load_lob_json(fixed_json_string)

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
        return load_lob_json(fixed_json_string)

    for key, value in list(json_dict.items()):
        if not value['bids'] or not value['asks']:
            del json_dict[key]

    return json_dict


def get_trade_data(pair, date_start, date_end, frequency = timedelta(seconds=10)):
    '''
    Function that returns a dataframe of resampled trade data and ready
    to be concatenated to a quotes dataframe with depth (Level = -1)

    Arguments:
    pair -- string, curency pair to return (e.g.'USDT_BTC')
    date_start -- string, timeseries start
    date_end -- string, timeseries end
    frequency -- timedelta, the minimum time granularity (e.g. timedelta(seconds=10))
    '''

    print(f'Checking for cached trade data from {date_start} to {date_end}')

    configuration = config()
    raw_data_folder = configuration['folders']['raw_trade_data']
    resampled_data_folder = configuration['folders']['resampled_data']

    date_start = datetime.strptime(date_start, '%Y-%m-%d')
    date_end = datetime.strptime(date_end, '%Y-%m-%d')
    freq = f'{int(frequency.total_seconds())}s'
    os.makedirs(f'{resampled_data_folder}/{pair}/trades/{freq}', exist_ok=True)

    data = []

    # Loop through day folders
    date_to_process = date_start
    while date_to_process <= date_end:
        resampled_file_path = f'{resampled_data_folder}/{pair}/trades/{freq}/{datetime.strftime(date_to_process, "%Y-%m-%d")}.csv.gz'
        if os.path.isfile(resampled_file_path):
            print(f'Found {resampled_file_path}')
        else:
            print(f'Generating {resampled_file_path}')
            raw_file_name = f'{pair}-{datetime.strftime(date_to_process, "%Y%m%d")}.csv.gz'
            raw_file_path = f'{raw_data_folder}/{pair}/{raw_file_name}'
            if not os.path.isfile(raw_file_path):
                s3_resource = boto3.resource('s3')

                """
                The calls to AWS STS AssumeRole must be signed with the access key ID and secret access key of an existing IAM user.
                The credentials can be in environment variables or in a configuration file and will be discovered automatically by the boto3.client() function.
                For more information, see the Python SDK documentation: http://boto3.readthedocs.io/en/latest/reference/services/sts.html#client
                """
                if configuration['other'].getboolean('cross_account_access'):
                    sts_client = boto3.client('sts')
                    response = sts_client.assume_role(RoleArn=configuration['other']['cross_account_access_role'], RoleSessionName="AssumeRoleSession")
                    s3_resource = boto3.resource(
                        's3',
                        aws_access_key_id = response['Credentials']['AccessKeyId'],
                        aws_secret_access_key = response['Credentials']['SecretAccessKey'],
                        aws_session_token = response['Credentials']['SessionToken'],
                    )

                trade_data_bucket = s3_resource.Bucket(configuration['buckets']['trade_data'])
                trade_data_bucket.download_file(f'{pair}/{raw_file_name}', f'{raw_file_path}')
                print(f'Downloaded {raw_file_name} from S3')

            day_data = pd.read_csv(raw_file_path, parse_dates=['date'])

            #df_trades['date'] = pd.to_datetime(df_trades['date'])

            df_trades_grp = day_data.groupby([pd.Grouper(key='date', freq=freq, dropna=False), 'type']).agg({'amount':'sum', 'rate':'mean'}).reset_index()
            df_trades_piv = df_trades_grp.pivot(values=['amount', 'rate'], columns='type',index='date').reset_index()

            df_trades_piv.columns = list(map("_".join, df_trades_piv.columns)) # "flatten" column names
            df_trades_piv.rename(columns={'date_':'Datetime', 'amount_buy':'Ask_Size', 'amount_sell':'Bid_Size', 'rate_buy':'Ask_Price', 'rate_sell':'Bid_Price'}, inplace=True)

            # fill gaps with no trades
            date_range_reindex = pd.DataFrame(pd.date_range(df_trades_piv['Datetime'].min(), df_trades_piv['Datetime'].max(), freq=freq), columns=['Datetime'])
            df_trades_piv = pd.merge(df_trades_piv, date_range_reindex, right_on='Datetime', left_on='Datetime', how='right')

            # impute NAs - zero for size and last px for price
            df_trades_piv.loc[:,['Ask_Size', 'Bid_Size']] = df_trades_piv.loc[:,['Ask_Size', 'Bid_Size']].fillna(0)
            df_trades_piv.loc[:,['Ask_Price', 'Bid_Price']] = df_trades_piv.loc[:,['Ask_Price', 'Bid_Price']].fillna(method='ffill')

            # level -1 to keep it separate from order book depth
            df_trades_piv['Level'] = -1
            df_trades_piv.to_csv(resampled_file_path, compression='gzip')

        date_to_process += timedelta(days=1) # the most nested folder is a day of the month 
        data.append(resampled_file_path)

    return dd.read_csv(data, compression='gzip')


def cnn_data_reshaping(X, Y, T):
    '''
    Reshape/augment data for 1D convolutions
    Inputs: X -> np.array with shape (lentgh_timeseries, # entries * order book depth for each timestamp)
            Y -> np.array with shape (length timeseries, 1)
            T -> int: # past timesteps to augment each timestamp

    Output: reshaped X and Y

    To do: accomodate for 2D convs
    '''
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))

    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    dataX = dataX.reshape(dataX.shape + (1,)) # no need to add the extra dimension for 1d conv

    print(f'shape X:{dataX.shape}, shape Y:{dataY.shape}')

    return dataX, dataY


def reshape_lob_levels(z_df, output_type='array'):
    '''
    Reshape data in a format consistent with deep LOB paper
    '''

    reshaped_z_df = z_df.pivot(index='Datetime', 
                          columns='Level', 
                          values=['Ask_Price', 'Ask_Size', 'Bid_Price', 'Bid_Size']).T.reset_index()\
                          .sort_values(by=['Level', 'level_0'], ascending=[True, True])\
                          .set_index(['Level', 'level_0']).T

    dt_index = reshaped_z_df.index

    print(f'Depth Values shape: {reshaped_z_df.shape}')
    print(f'Datetime Index shape: {dt_index.shape}')

    if output_type == 'dataframe':

        return reshaped_z_df, dt_index
        
    elif output_type == 'array':

        depth_values = reshaped_z_df.values # numpy array ready to be used as input for cnn_data_reshaping
        return depth_values, dt_index


# Evaluate model predictions
def back_to_labels(x):
    '''Map ternary predictions in format [0.01810119, 0.47650802, 0.5053908 ]
    back to original labels 1,0,-1. Used in conjuction with numpy.argmax, which 
    returns the index of the label with the highest probability.
    '''

    if x == 0:
        return 0

    elif x == 1:
        return 1

    elif x == 2:
        return -1

frequency = timedelta(seconds=60)
pair = 'USDT_BTC'
date_start = '2021-03-14'
date_end = '2021-03-15'
lob_depth = 10
norm_type = 'dyn_z_score'
roll = 7200 * 6
label_technique = 'three_steps'

train_dyn_df, test_dyn_df, top_ob_train, top_ob_test = import_px_data(frequency, pair, date_start, date_end, lob_depth, norm_type, roll)