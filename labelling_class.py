import numpy as np
import pandas as pd
import scipy.signal
from sklearn.preprocessing import MinMaxScaler


class Labels_Generator:

    def __init__(self, mid_px):
        self.mid_px = mid_px


    def get_smooth_px(self):
        ''' Smoothed mid price signal '''

        # smooth prices - Savitzky–Golay filter
        smooth_px = pd.Series(scipy.signal.savgol_filter(self.mid_px, 31, 1))
        return smooth_px


    def get_norm_smooth_px(self):
        ''' Min max scaler to normalize smoothed signal '''

        # scale smoothed time series (squash between 0 and 1 with min max scaler)
        values = self.get_smooth_px().values.reshape(self.mid_px.shape[0],1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(values)
        norm_smooth_px = pd.Series(scaler.transform(values).reshape(values.shape[0]))
        return norm_smooth_px


    def get_raw_labels(self):
        ''' Logic to generate basic labels from normalized smoothed px '''

        # first level difference - smoothed time series direction
        a = 0
        # gradient preserves shape, default unitary spacing
        d = np.gradient(self.get_norm_smooth_px())

        # label based on the direction
        self.labels = pd.Series(np.where(d>a, 1, np.where(d<a, -1, 0)),  name='labels')


    def get_cleaned_labels(self, fillna_value=None, fillna_method=None, **kwargs):
        ''' Labels cleaning. Can be reused multiple times  specifying the fillna value OR method and 
        passing the criteria to filter the trade dataframe as kwargs arguments

        fillna_value: Pandas "value" arg for fillna - applied to clean labels - usually 0 if not None
        fillna_method: Pandas "method" arg for fillna - applied to clean labels - usually "ffill" if not None
        **kwargs: trades dataframe columns name with relative threshold to filter for (<=). Example: gross_returns=0.002
        '''

        # recap dataframe
        df_trades_columns = ['trade_grouper', 'labels', 'trade_len', 'gross_returns']
        df_trades = get_strategy_pnl(self.mid_px, self.labels)[df_trades_columns].dropna(subset=['gross_returns'])

        # locate short unprofitable labels, replace them with NAs and fill them with prev label values
        df_trades['cleaned_labels'] = df_trades['labels']
        # build df query with keyward passed to locate index of unprofitable labels
        query = ' & '.join([f'`{k}`<={v}' for k, v in kwargs.items()])
        print(f'Criteria {query}')
        df_trades.loc[df_trades.query(query).index, 'cleaned_labels'] = pd.NA
        # fillna methodology depends on the args passed to the function
        df_trades['cleaned_labels'].fillna(value=fillna_value, method=fillna_method, inplace=True)

        # expand table trades - one row per trade -  back into a full labels timeseries
        cleaned_labels = np.empty(self.labels.shape[0])
        cleaned_labels[:] = np.nan
        cleaned_labels = pd.Series(cleaned_labels, name='cleaned_labels')
        cleaned_labels.loc[df_trades['trade_grouper']] = df_trades['cleaned_labels']
        # ffill the 'exploded' timeseries with prev values
        self.labels = cleaned_labels.fillna(method='ffill')

        return df_trades

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


def cleaned_labels(target_timeseries, method='three_steps', print_details=True):
    '''
    Wrapper that execute all the steps for a given method
    target_timeseries -- pandas series
    '''
    labels_gen = Labels_Generator(target_timeseries)

    if method == 'three_steps':

        #step 1
        labels_gen.get_raw_labels()
        if print_details:
            print('\n##### Step 1 #####')
            label_insights(labels_gen.labels)

        # step 2 - first cleaning
        _ = labels_gen.get_cleaned_labels(fillna_method='ffill', gross_returns=0.005, trade_len=20)
        if print_details:
            print('\n##### Step 2 #####')
            label_insights(labels_gen.labels)

        # step 3 - second cleaning
        df_trades = labels_gen.get_cleaned_labels(fillna_value=0, gross_returns=0.005, trade_len=30)#, gross_returns=0.002)
        if print_details:    
            print('\n##### Step 3 #####')
            label_insights(labels_gen.labels)

        labels = labels_gen.labels

        return labels, labels_gen.get_smooth_px(), df_trades

    else:
        raise ValueError(f"Method {method} not recognized")



def three_barrier_labelling(mix_px, h=700, factor=[1.0020, 0.9980]):
    '''
    Alternative labelling technique inspired to "three barriers method" on Advances in Financial Machine Learning book
    explanation: https://mlfinlab.readthedocs.io/en/latest/labeling/tb_meta_labeling.html
    Compared to the method above it tends to lag. The current implementation can cut positive running labels at any point.
    barriers depending on volatility still to be implmeneted. Also vertical barrier should depend on volatility
    '''

    counter = 0
    output =  mix_px.copy(deep=True) ## can use smoothed verison here
    output = pd.DataFrame(output)
    output['labels'] = 12345 # placeholder

    while counter <= mix_px.shape[0]: ##
        print(f'start {counter} and end {counter+h}')
        slice_df = mix_px[counter:counter+h] # path prices ##
        ut = pd.Series(np.full((slice_df.shape[0], ), 1, dtype=float), index=slice_df.index) # upper threshold, to do: function of volatility
        lt = pd.Series(np.full((slice_df.shape[0], ), 1, dtype=float), index=slice_df.index) # lower threshold, to do: function of volatility

        take_profit = (ut*factor[0])-1 # upper barrier
        stop_loss = (lt*factor[1])-1  # lower barrier

        slice_df = (slice_df/slice_df[0]-1) # path returns

        take_profit_touch = slice_df[slice_df>take_profit].index.min() # find upper touch time
        stop_loss_touch = slice_df[slice_df<stop_loss].index.min() # find lower touch time

        if pd.isna(take_profit_touch) and pd.isna(stop_loss_touch):
            output.loc[slice_df.index[0]: slice_df.index[-1],'labels'] = 0 # if no touch, assign 0 till vertical barrier
            counter+=h

        elif not pd.isna(take_profit_touch) and pd.isna(stop_loss_touch):
            output.loc[slice_df.index[0]: take_profit_touch,'labels'] = 1 # assign 1 until take profit is touched
            counter+=slice_df.index.get_loc(take_profit_touch)

        elif pd.isna(take_profit_touch) and not pd.isna(stop_loss_touch):
            output.loc[slice_df.index[0]: stop_loss_touch,'labels'] = -1 # assign -1 until stop loss is touched
            counter+=slice_df.index.get_loc(stop_loss_touch)

        elif not pd.isna(take_profit_touch) and not pd.isna(stop_loss_touch): # assign +1 or -1 to what is touched first
            if take_profit_touch<stop_loss_touch:
                output.loc[slice_df.index[0]: take_profit_touch,'labels'] = 1
                counter+=slice_df.index.get_loc(take_profit_touch)

            elif stop_loss_touch<take_profit_touch:
                output.loc[slice_df.index[0]: take_profit_touch,'labels'] = -1
                counter+=slice_df.index.get_loc(stop_loss_touch)
        else:
            print(f'Occurrence not capture between {counter} and {counter+h}')

    return output