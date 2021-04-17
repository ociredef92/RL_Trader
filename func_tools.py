import numpy as np
import pandas as pd
from datetime import datetime
import os
from os import listdir
from os.path import isfile, join

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

def import_px_data(experiments_folder, frequency, pair, date_start, date_end, lob_depth, norm_type, roll):
    '''
    Function that loads preprocessed data ready to be shaped/used for the model to train.
    Experiment folder is the path where data has been cached. The other parameters are part of the
    unique cached file nomenclature. If the file does not exist, it is generated frrom the input data
    in the "else" block

    Arguments:
    experiments_folder -- string, path where standardized and compact non-standardized data is stored
    frequency --  timedelta, the minimum time granularity (e.g. timedelta(seconds=10))
    pair -- string, pair to be uploaded (e.g.'USDT_BTC')
    date_start -- string, timeseries start
    date_end -- string, timeseries end
    lob_depth -- integer, how many levels of the order book to be considered
    norm_type -- string, can assume values of 'z' or 'dyn' for z-score or dynamic z-score
    roll -- integer, function of the granularity provided
    '''

    cache_folder = f'{experiments_folder}/cache' # get ready data cache path
    frequency_seconds = int(frequency.total_seconds())
    os.makedirs(f'{cache_folder}/{pair}', exist_ok=True)

    # Data import - needs to be adjusted importing from several files using Dask
    quotes_file_name = f'{pair}--{lob_depth}lev--{frequency_seconds}sec--{date_start}--{date_end}.csv.gz'

    normalized_train_file = f'{cache_folder}/{pair}/TRAIN--{norm_type}-{roll}--{quotes_file_name}'
    normalized_test_file = f'{cache_folder}/{pair}/TEST--{norm_type}-{roll}--{quotes_file_name}'

    top_ob_train_file = f'{cache_folder}/{pair}/TRAIN_TOP--{quotes_file_name}'
    top_ob_test_file = f'{cache_folder}/{pair}/TEST_TOP--{quotes_file_name}'

    # normalized test file contains both trades and quotes
    if os.path.isfile(normalized_test_file): # testing for one of cache files, assuming all were saved
        # Import cached standardized data
        print(f'Reading cached {normalized_train_file}')
        train_dyn_df = pd.read_csv(normalized_train_file, index_col=1)
        print(f'Reading cached {normalized_test_file}')
        test_dyn_df = pd.read_csv(normalized_test_file, index_col=1)

        print(f'Reading cached {top_ob_train_file}')
        top_ob_train = pd.read_csv(top_ob_train_file, index_col=[0,1])
        print(f'Reading cached {top_ob_test_file}')
        top_ob_test = pd.read_csv(top_ob_test_file, index_col=[0,1])

    else: # check separately for quotes and trades input files
        input_data_folder = f'{experiments_folder}/input' # non standardized input data

        ## check if quotes file exist in input folder
        if os.path.isfile(f'{input_data_folder}/{quotes_file_name}'):
            print(f'Reading {input_data_folder}/{quotes_file_name}')
            data = pd.read_csv(f'{input_data_folder}/{quotes_file_name}', index_col=0)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            assert lob_depth == data['Level'].max() + 1 # number of levels of order book - maybe add extra + 1 for trades

        
        else:
            # create non standardized input file using LOB data
            # this block will return data just created from LOBData
                pass

        # check if input file for trades file exists
        trades_file_name = f'{pair}-trades-{frequency_seconds}sec-{date_start}-{date_end}.csv.gz'
        if os.path.isfile(f'{input_data_folder}/{trades_file_name}'):
            print(f'Reading {input_data_folder}/{trades_file_name}')
            trades_data = pd.read_csv(f'{input_data_folder}/{trades_file_name}')
            trades_data['date'] = pd.to_datetime(trades_data['date'])

        else:
            # create non standardized input trades file using the function below
            # this block will return data just created from LOBData

        # once input files have been correctly read from the input folder, it's time to create a single standardized cache for trades and quotes
        train_dyn_df, test_dyn_df, top_ob_train, top_ob_test = standardized_data_cache(data, roll, lob_depth, normalized_train_file, normalized_test_file, top_ob_train_file, top_ob_test_file)

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


def standardized_data_cache(data, roll, lob_depth, normalized_train_file, normalized_test_file, top_ob_train_file, top_ob_test_file):
    # Train test split
    train_test_split = int((data.shape[0] / lob_depth) * 0.7) # slice reference for train and test
    train_timestamps = data['Datetime'].unique()[:train_test_split]
    test_timestamps = data['Datetime'].unique()[train_test_split:]

    train_cached_data = data[data['Datetime'].isin(train_timestamps)].set_index(['Datetime', 'Level'])
    test_cached_data = data[data['Datetime'].isin(test_timestamps)].set_index(['Datetime', 'Level'])

    ### trades managed here before standardization


    print(f'Train dataset shape: {train_cached_data.shape} - Test dataset shape: {test_cached_data.shape}')

    roll_shift = roll+1 # rolling period for dyn z score - + 1 from shift in ft.normalize

    # Training
    # custom rolling standardization for px and size separately
    train_dyn_prices = standardize(train_cached_data[['Ask_Price', 'Bid_Price']], lob_depth, 'dyn_z_score', roll)
    train_dyn_volumes = standardize(train_cached_data[['Ask_Size', 'Bid_Size']], lob_depth, 'dyn_z_score', roll)
    train_dyn_df = pd.concat([train_dyn_prices, train_dyn_volumes], axis=1).reset_index() # concat along row index #1
    print(f'Saving {normalized_train_file}')
    train_dyn_df.to_csv(normalized_train_file, compression='gzip') # save normalized data to csv 

    top_ob_train = train_cached_data[train_cached_data.index.get_level_values(1)==0][roll_shift:] #3
    top_ob_train['Mid_Price'] = (top_ob_train['Ask_Price'] + top_ob_train['Bid_Price']) / 2
    top_ob_train['Spread'] = (top_ob_train['Ask_Price'] - top_ob_train['Bid_Price']) / top_ob_train['Mid_Price']
    top_ob_train['merge_index'] = top_ob_train.reset_index().index.values # useful for merging later
    print(f'Saving {top_ob_train_file}')
    top_ob_train.to_csv(top_ob_train_file, compression='gzip') # save top level not normalized to csv

    # print(f'Saving {normalized_data_folder}/{pair}/TRAIN_top--{norm_type}-{roll}--{input_file_name}')
    # train_dyn_df[train_dyn_df['Level']==0].to_csv(f'{normalized_data_folder}/{pair}/TRAIN_TOP--{norm_type}-{roll}--{input_file_name}', compression='gzip') # save top level to csv 

    # Test
    # custom rolling standardization for px and size separately
    test_dyn_prices = standardize(test_cached_data[['Ask_Price', 'Bid_Price']], lob_depth, 'dyn_z_score', roll)
    test_dyn_volumes = standardize(test_cached_data[['Ask_Size', 'Bid_Size']], lob_depth, 'dyn_z_score', roll)
    test_dyn_df = pd.concat([test_dyn_prices, test_dyn_volumes], axis=1).reset_index() # concat along row index #2
    print(f'Saving {normalized_test_file}')
    test_dyn_df.to_csv(normalized_test_file, compression='gzip') # save normalized data to csv

    top_ob_test = test_cached_data[test_cached_data.index.get_level_values(1)==0][roll_shift:] #4
    top_ob_test['Mid_Price'] = (top_ob_test['Ask_Price'] + top_ob_test['Bid_Price']) / 2
    top_ob_test['Spread'] = (top_ob_test['Ask_Price'] - top_ob_test['Bid_Price']) / top_ob_test['Mid_Price']
    top_ob_test['merge_index'] = top_ob_test.reset_index().index.values # useful for merging later
    print(f'Saving {top_ob_test_file}')
    top_ob_test.to_csv(top_ob_test_file, compression='gzip') # # save top level not normalized to csv

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


def fetch_s3_trade_files(s3_folder, input_data_folder, pair, frequency):
    s3_trades_folder = f'{s3_folder}/trades'

    # series with dates of all the files present in the folder
    trade_files = pd.Series([datetime.strptime(f[:-7][-8:], '%Y%m%d') for f in listdir(f'{s3_trades_folder}/{pair}') if isfile(join(f'{s3_trades_folder}/{pair}', f))])

    # create date range from newest to oldedst trade file
    dates = pd.date_range(start=trade_files.min(), end=trade_files.max(), freq='D').strftime(date_format='%Y%m%d')

    # read trades
    print('reading trade files')
    df_trades = pd.DataFrame([])
    for date in dates:
        df_trades = pd.concat([df_trades, pd.read_csv(f'{s3_trades_folder}/{pair}/{pair}-{date}.csv.gz')])

    destination_path = f"{input_data_folder}/{pair}-trades-{int(frequency.total_seconds())}s-{trade_files.min().strftime(format='%Y_%m_%d')}-{trade_files.max().strftime(format='%Y_%m_%d')}.csv.gz"
    print(f'saving file at {destination_path}')
    df_trades.to_csv(destination_path, compression='gzip')
    return df_trades


def process_s3_trade_files():
    # 
    # df_trades['date'] = pd.to_datetime(df_trades['date'])
    # df_trades_grp = df_trades.groupby([pd.Grouper(key='date', freq='10s', dropna=False), 'type']).agg({'amount':'sum', 'rate':'mean'}).reset_index()
    # df_trades_piv = df_trades_grp.pivot(values=['amount', 'rate'], columns='type',index='date').reset_index()

    # df_trades_piv.columns = list(map("_".join, df_trades_piv.columns)) # "flatten" column names
    # df_trades_piv.rename(columns={'date_':'Datetime', 'amount_buy':'Ask_Size', 'amount_sell':'Bid_Size', 'rate_buy':'Ask_Price', 'rate_sell':'Bid_Price'}, inplace=True)

    # # fill gaps with no trades
    # date_range_reindex = pd.DataFrame(pd.date_range(df_trades_piv['Datetime'].min(), df_trades_piv['Datetime'].max(), freq="10s"), columns=['Datetime'])
    # df_trades_piv = pd.merge(df_trades_piv, date_range_reindex, right_on='Datetime', left_on='Datetime', how='right')

    # # impute NAs - zero for size and last px for price
    # df_trades_piv.loc[:,['Ask_Size', 'Bid_Size']] = df_trades_piv.loc[:,['Ask_Size', 'Bid_Size']].fillna(0)
    # df_trades_piv.loc[:,['Ask_Price', 'Bid_Price']] = df_trades_piv.loc[:,['Ask_Price', 'Bid_Price']].fillna(method='ffill')


    # df_trades_piv['Level'] = -1
    pass

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



# Evaluate model preditctions
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



