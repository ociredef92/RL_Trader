import numpy as np
import pandas as pd
import plotly_express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


# Visualization
def plot_labels_line(px_ts, labels, title='Labels', **kwargs):
    '''Plot labels against price.
    Takes two pandas timeseries as inputs. These need to be subsets of the same
    DataFrame or have same length
    '''
    #print(kwargs)
    # check index
    condition = (px_ts.index == labels.index).sum()
    assert condition == px_ts.shape[0] == labels.shape[0], 'px_ts and labels must have the same index to be correctly plotted'

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=px_ts, x=px_ts.index, name='Price'), secondary_y=False)
    fig.add_trace(go.Scatter(y=labels, x=labels.index, name='Labels', marker=dict(color='rgba(240, 52, 52, 0.3)')), 
        secondary_y=True)


    for arg, key in zip(kwargs.values(), kwargs.keys()):
        if 'label' in key or 'direction' in key:
            fig.add_trace(go.Scatter(y=arg, x=arg.index, name=key), secondary_y=True)
        else:
            fig.add_trace(go.Scatter(y=arg, x=arg.index, name=key), secondary_y=False)

    fig.update_layout(title=f'<b>{title}</b>', width=1200, height=800)
    fig.update_yaxes(title_text='ccy', fixedrange= False, secondary_y=False)
    fig.update_yaxes(title_text='label', secondary_y=True)
    #fig.update_yaxes(fixedrange= False)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
        )
    )

    return fig.show()


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

# Backtesting Visualization
def plot_trades_distribution(df_trades, bin_size=0.0001, metric='gross_returns', fig_width=900, fig_height=550):
    ''' Plot trades distribution and approx distribution curve.
    Takes as an input df_trades from stratgy pnl, bin_size (default 1bp) and gross return as metrix
    '''

    trades = [df_trades['gross_returns'].values]
    labels = ['Trades']
    fig = ff.create_distplot(trades, labels, bin_size = 0.0001, show_rug=False)
    # Add shapes
    avg = np.mean(trades)
    stdev = np.std(trades)

    fig.add_shape(type="line", yref='paper',
        x0=avg, y0=0, x1=avg, y1=1,
        line=dict(color="RoyalBlue",width=2)
    )

    fig.add_shape(type="line", yref='paper',
        x0=avg+stdev, y0=0, x1=avg+stdev, y1=1,
        line=dict(color="RoyalBlue",width=2, dash="dot")
    )

    fig.add_shape(type="line", yref='paper',
        x0=avg-stdev, y0=0, x1=avg-stdev, y1=1,
        line=dict(color="RoyalBlue",width=2, dash="dot")
    )

    fig.add_shape(type="line", yref='paper',
        x0=0, y0=0, x1=0, y1=1,
        line=dict(color="rgba(0, 0, 0, 0.5)",width=2, dash="dashdot")
    )

    fig.update_layout(title=f"<b>Trades distribution - {metric}</b>", width=fig_width, height=fig_height, xaxis=dict(tickformat=',.3%'))
    fig.show()    


def plot_trades_length_overview(df_trades, x='trade_len',  y='gross_returns'):
    ''' Plot visual insight for lavels on x variable (default "trade_len"):
    1) histogram with count of x
    2) histogram with x vs average y (default "gross_returns")
    3) individual trades x vs y

    Takes as an input df_trades from stratgy pnl, with x and y being columns of df_trades
    '''

    max_trade_length = int(df_trades['trade_len'].max())
    hist_trade_length = px.histogram(df_trades, x=x, color='labels', title=f'<b>{x}</b>')
    avg = df_trades['trade_len'].mean() # average trade length
    hist_trade_length.add_shape(type="line", yref='paper',
        x0=avg, y0=0, x1=avg, y1=1,
        line=dict(color="rgba(0, 0, 0, 0.5)",width=2, dash="dashdot")
    )
    hist_trade_length.show()

    # Plot net returns (by length and average returns)
    hist_ret_len = px.histogram(df_trades, x=x, y=y, histfunc='avg', color='labels', nbins=max_trade_length, title=f'<b>{y} by {x}</b>')
    hist_ret_len.update_layout(yaxis=dict(tickformat=',.3%'))
    hist_ret_len.show()

    # Plot individual trades vs trade length
    avg_net_by_length = df_trades.groupby('trade_len')['gross_returns'].mean()
    ret_len_scatter = px.scatter(df_trades, x=x, y=y, color=df_trades['labels'].astype('str'), opacity=0.3, title=f'<b>{y} single trades</b>')
    ret_len_scatter.add_trace(go.Scatter(x=avg_net_by_length.index, y=avg_net_by_length.values, mode='lines', name='Average'))
    ret_len_scatter.update_layout(yaxis=dict(tickformat=',.3%'))
    ret_len_scatter.show()


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



# OLD functions to DELETE

# # Model training - labelling and backtesting
# def get_labels(ts, span_plus, span_minus, a, technique='ma', long_only=True, return_smooth=False):
#     '''
#     Function to label timeseries - buy, sell, nothing

#     Arguments:
#     ts -- pandas series or array
#     span_plus -- integer, prediction horizon (how much forward am I looking to get price direction)
#     span_minus -- integer, prediction horizon (how much backward am I looking to get price direction)
#     alpha -- float, threshold for applying labels
#     long_only -- boolean that to turn off/on profits from short-selling

#     Returns: pandas series
#     '''

#     if technique == 'ma':
#         m_minus = ts.rolling(span_minus).mean() # mean prev k prices
#         m_plus = ts.shift(-span_plus).rolling(span_plus).mean() # # mean next k prices

#     elif technique == 'ema':
#         m_minus = ts.ewm(span=span_minus, min_periods=span_minus).mean()
#         m_plus = ts.shift(-span_plus).ewm(span=span_plus, min_periods=span_plus).mean()

#     # direction of price movements at time t
#     #direction = (m_plus - m_minus) / m_minus

#     direction_pos = (m_plus - m_minus) / m_minus
#     direction_neg = (m_minus - m_plus) / m_plus
#     direction = pd.Series(np.where(ts>=0, direction_pos, direction_neg)) # flip when ts is neg

#     if long_only:
#         # assign labels based on alpha threshold
#         return pd.Series(np.where(direction>a, 1, 0), index=direction.index, name='labels')
    
#     else:
#         if return_smooth:
#             return pd.Series(np.where(direction>a, 1, 
#                 np.where(direction<-a, -1, np.nan)), index=direction.index, name='labels'), m_minus, m_plus    
#         else:
#             # assign labels based on alpha threshold
#             return pd.Series(np.where(direction>a, 1, 
#                 np.where(direction<-a, -1, np.nan)), index=direction.index, name='labels')


# def get_pnl(px_ts, labels, trading_fee=0.000712):
#     '''Function to get pnl from a price time series and respective labels
#     px_ts and labels series must be mergeable by index

#     Arguments:
#     px_ts -- pandas series or array with datetime index
#     labels -- pandas series or array with datetime index. values: 1 buy, 0 nothing, -1 sell
#     trading_fee -- floating express in decimal form

#     Returns a pandas series with time index, np array with label change index
#     '''
    
#     df = pd.merge(px_ts, labels, left_index=True, right_index=True)
#     df.columns=['px', 'labels']
    
#     labels_ext = np.concatenate(( [0], labels.values, [0])) # extend array for comparison
#     idx = np.flatnonzero(labels_ext[1:] != labels_ext[:-1])
#     #print(idx.shape[0]) # how many transactions
#     if idx[-1] >= df.shape[0]:
#         idx = idx[:-1] # non zero indices - remove last. Avoid errors when transaction occurs on last label


#     tr_fees = np.ones((labels.shape[0]))
#     tr_fees[idx] = 1 - trading_fee

#     df['return'] = df['px'].pct_change()
#     df['realized_return'] = df['return'] * df['labels']
#     df['trade_flag'] = df.index.isin(idx)
#     df['pnl'] = ((df['labels'] * df['return']) + 1).cumprod()

#     return ((df['labels'] * df['return']) + 1).cumprod() - 1, df, idx # labels and label change index

# def plot_labels(labels, y0=0.46):
      ##too SLOW
#     label_change = labels - labels.shift(1)
#     label_change = label_change[label_change!=0]
#     background_color = [
#                         dict(
#                             type="rect",
#                             # x-reference is assigned to the x-values
#                             xref="x1",
#                             # y-reference is assigned to the plot paper [0,1]
#                             yref="paper",
#                             x0=label_change.index[i],
#                             y0=y0,
#                             x1=label_change.index[i+1],
#                             y1=1,
#                             fillcolor="Red",
#                             opacity=0.2,
#                             layer="below",
#                             line_width=0,
#                         )   

#                         if int(labels[labels.index == label_change.index[i]])==-1 or int(labels[labels.index == label_change.index[i]])==-2

#                             else
#                                 (
#                                 dict(
#                                     type="rect",
#                                     # x-reference is assigned to the x-values
#                                     xref="x",
#                                     # y-reference is assigned to the plot paper [0,1]
#                                     yref="paper",
#                                     x0=label_change.index[i],
#                                     y0=y0,
#                                     x1=label_change.index[i+1],
#                                     y1=1,
#                                     fillcolor="Green",
#                                     opacity=0.2,
#                                     layer="below",
#                                     line_width=0,
#                                 )   


#                                 if int(labels[labels.index == label_change.index[i]])==1 or int(labels[labels.index == label_change.index[i]])==2

#                                     else
#                                         dict(
#                                             type="rect",
#                                             # x-reference is assigned to the x-values
#                                             xref="x",
#                                             # y-reference is assigned to the plot paper [0,1]
#                                             yref="paper",
#                                             x0=label_change.index[i],
#                                             y0=y0,
#                                             x1=label_change.index[i+1],
#                                             y1=1,
#                                             fillcolor="#E5ECF6", # default plotly background
#                                             opacity=0.2,
#                                             layer="below",
#                                             line_width=0,
#                                         )   
#                                 )

#                             for i in range(len(label_change)-1)
#     ]
#     return background_color

# Get data in the desired format - similar to deep lob
# Steps: pivot data, transpose and reset index (easier to sort than columns),
# sort by order book level and then event type in order to have at each level
# ask_price, ask_size, bid_price, bid_size. Finally reset the index back and transpose


#   def get_strategy_pnl(px_ts, labels, trading_fee=0.000712, min_profit=0.0020, plotting=False, return_df=True):
    
#     df = pd.merge(px_ts, labels, left_index=True, right_index=True)
#     df.columns = ['px', 'labels']

#     labels_ext = np.concatenate(( [0], labels.values, [0])) # extend array for comparison
#     idx = np.flatnonzero(labels_ext[1:] != labels_ext[:-1]) # verify if alignement is correct
#     # non zero indices - remove last. Avoid errors when transaction occurs on last label
#     if idx[-1] >= df.shape[0]:
#         idx = idx[:-1] 

#     #df['pctg_chg'] = df['px'].pct_change()
#     df['log_ret'] = np.log(df['px']) - np.log(df['px'].shift(1))
#     df['individual_return'] = df['log_ret'] * df['labels']# need to add +1 to multiply with tr feee

#     # auxiliary column to perform groupby and get rough profit estimate
#     df['trade_grouper'] = np.nan
#     df['trade_grouper'].loc[idx] = idx
#     df['trade_grouper'] = df['trade_grouper'].fillna(method='ffill')

#     # calculate profits
#     trade_gross_profit = pd.Series(df.groupby('trade_grouper')['individual_return'].sum(), name='Gross returns') # each grouper represents a trade
#     positive_trades = pd.Series(trade_gross_profit[trade_gross_profit - min_profit > 0], name='individual_positive_returns')

#     df = pd.merge(df, trade_gross_profit, left_index=True, right_index=True, how='outer') # add gross
#     df = pd.merge(df, positive_trades, left_index=True, right_index=True, how='outer') # add pos trades to the df

#     profit = df['individual_positive_returns'].fillna(0).sum()

#     #
#     print(f'''Total trades: {trade_gross_profit.shape[0]}, # trades > {min_profit}: {positive_trades.shape[0]}, profit: {profit}''')

#     if plotting:
#         histo_trades = px.histogram(positive_trades)
#         histo_trades.show()
#         cum_profit = px.line(df['individual_positive_returns'].fillna(0).cumsum().iloc[::1000])
#         cum_profit.show()
#     if return_df:
#         # create cleaned labels column - wasteful to run this on optimization stage
#         positive_trade_idx = df[df['individual_positive_returns']>0]['trade_grouper'] # positive trade start
#         positive_df_idx = df[df['trade_grouper'].isin(positive_trade_idx)].index # all "timeseries" of positive trades
#         df['cleaned_labels'] = 0 # create column with all 0 labels
#         df['cleaned_labels'].loc[positive_df_idx] = df['labels'].loc[positive_df_idx] # replace 0s with labels of positive trades
#         return profit, df
#     else:
#         return profit

