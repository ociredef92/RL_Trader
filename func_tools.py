import numpy as np
import pandas as pd


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


# Model training - data preparation
def standardize(ts, ob_levels,norm_type='z_score', roll=0):
    '''
    Function to standardize (mean of zero and unit variance) timeseries

    Arguments:
    ts -- pandas series or df having timestamp and ob level as index to allow sorting (dynamic z score)
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

    

def label_insights(labels):
    '''
    Take a np.array of labels as an input and return
    insights into number of labels, transactions, imbalance
    labels has to be one dimentional ie: (labels.shape, )
    '''

    # get for how long labels are "in the market"
    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    percent_labels = counts_labels / counts_labels.sum()
    a_ext = np.concatenate(( [0], labels.values, [0])) # extend array for comparison
    idx = np.flatnonzero(a_ext[1:] != a_ext[:-1]) # non zero indices - transactions

    print(f'Labels shape: {labels.shape}')
    print(f'Labels: {unique_labels} \nCount: {counts_labels} \nPctg: {percent_labels}')
    print(f'Number of trades: {idx.shape[0]}')

    return idx.shape[0]


# Backtesting
def get_strategy_pnl(px_ts, labels):
    
    df = pd.merge(px_ts, labels, left_index=True, right_index=True) # default how is inner
    df.index = np.arange(0,df.shape[0])
    df.columns = ['px', 'labels']

    labels_ext = np.concatenate(( [0], labels.values, [0])) # extend array for comparison
    idx = np.flatnonzero(labels_ext[1:] != labels_ext[:-1]) # verify if alignement is correct
    # non zero indices - remove last. Avoid errors when transaction occurs on last label
    if idx[-1] >= df.shape[0]:
        idx = idx[:-1] 

    df['log_ret'] = np.log(df['px']) - np.log(df['px'].shift(1))#df['px'].pct_change()#np.log(df['px']) - np.log(df['px'].shift(1))
    df['individual_return'] = df['log_ret'] * df['labels']# need to add +1 to multiply with tr feee

    # auxiliary column to perform groupby and get rough profit estimate
    df['trade_grouper'] = np.nan
    df['trade_grouper'].loc[idx] = idx
    df['trade_grouper'] = df['trade_grouper'].fillna(method='ffill')

    # series with trade length filled across the df
    df['trade_len'] = df.groupby(['labels', 'trade_grouper'])['px'].transform('count')

    # calculate profits
    trade_gross_profit = pd.Series(df.groupby('trade_grouper')['individual_return'].sum(), name='gross_returns') # each grouper represents a trade
    # add gross returns at the beginning of each trade in df
    df = pd.merge(df, trade_gross_profit, left_index=True, right_index=True, how='outer')

    #average profit
    trades_df = df[df['labels']!=0]
    n_trades = trades_df['gross_returns'].count()
    tot_return = trades_df['gross_returns'].sum()
    avg_return = trades_df['gross_returns'].mean()

    print(f'''Total non zero trades: {n_trades}, sum of returns: {tot_return:.2f}, average return: {avg_return:.6f}''')


    # df['cleaned_labels'] = 0 # create column with all 0 labels
    # df['cleaned_labels'].loc[positive_df_idx] = df['labels'].loc[positive_df_idx] # replace 0s with labels of positive trades
    
    return df


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



