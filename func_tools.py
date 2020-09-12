import numpy as np
import pandas as pd


def normalize(ts, norm_type='z_score', roll=0):
    '''
    Function to normalize timeseries

    Arguments:
    ts -- pandas series or array
    norm_type -- string, can assume values of 'z' or 'dyn' for z-score or dynamic z-score
    roll -- integer, rolling window for dyanamic normalization.

    Returns: pandas series
    '''
    
    if norm_type=='z_score':
        
        if ts.shape[1] > 1:
            ts_stacked = ts.stack()
        else:
            ts_stacked = ts
        
        return (ts-ts_stacked.mean()) / ts_stacked.std()
    
    # dynamic can't accomodate multi columns normalization yet
    elif norm_type=='dyn_z_score' and type(roll) is int and roll>0:
        return (ts - ts.rolling(roll).mean().shift(1) 
              ) / ts.rolling(roll).std(ddof=0).shift(1)

    raise ValueError("Oops! Check your inputs and Try again...")


def get_labels(ts, k, alpha):
    '''
    Function to label timeseries - buy, sell, nothing

    Arguments:
    ts -- pandas series or array
    k -- integer, prediction horizon (how much back and forward am I looking to get price direction)
    alpha -- float, threshold for applying labels

    Returns: pandas series
    '''

    m_minus = ts.rolling(k).mean() # mean prev k prices
    m_plus = ts.shift(-k).rolling(k).mean() # # mean next k prices

    # direction of price movements at time t
    direction = (m_plus - m_minus) / m_minus

    # assign labels based on alpha threshold
    labels = direction.apply(lambda x: 1 if x>alpha else (-1 if x<-alpha else 0))

    return labels


def get_pnl(px_ts, labels, long_only=True):
    '''
    Function to get pnl from a price time series and respective labels

    Arguments:
    px_ts -- pandas series or array with datetime index
    labels -- pandas series or array with datetime index. values: 1 buy, 0 nothing, -1 sell
    long_only -- boolean that to turn off/on profits from short-selling

    Returns a pandas series with time index
    '''
    
    df = pd.merge(px_ts, labels, left_index=True, right_index=True)
    df.columns=['px', 'labels']
    
    if long_only:

        df['return'] = df['px'].pct_change() + 1

        return df[df['labels']==1]['return'].cumprod() - 1

    else:
        df['return'] = df['px'].pct_change() #+ 1
        
        df['ls_return'] = df.apply(lambda x: x['return'] if x['labels']==1 
                                   else (x['return']*-1 if x['labels']==-1 else 0), axis=1) + 1
        
        return df[(df['labels']==1) | (df['labels']==-1)]['ls_return'].cumprod() - 1



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

    dataX = dataX.reshape(dataX.shape)# + (1,)) # no need to add the extra dimension for 1d conv

    print(f'shape X:{dataX.shape}, shape Y:{dataY.shape}')

    return dataX, dataY


def plot_labels(labels):

    label_change = labels - labels.shift(1)
    label_change = label_change[label_change!=0]
    background_color = [
                        dict(
                            type="rect",
                            # x-reference is assigned to the x-values
                            xref="x1",
                            # y-reference is assigned to the plot paper [0,1]
                            yref="paper",
                            x0=label_change.index[i],
                            y0=0.44,
                            x1=label_change.index[i+1],
                            y1=1,
                            fillcolor="Red",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        )   

                        if int(labels[labels.index == label_change.index[i]])==-1 or int(labels[labels.index == label_change.index[i]])==-2

                            else
                                (
                                dict(
                                    type="rect",
                                    # x-reference is assigned to the x-values
                                    xref="x",
                                    # y-reference is assigned to the plot paper [0,1]
                                    yref="paper",
                                    x0=label_change.index[i],
                                    y0=0.44,
                                    x1=label_change.index[i+1],
                                    y1=1,
                                    fillcolor="Green",
                                    opacity=0.2,
                                    layer="below",
                                    line_width=0,
                                )   


                                if int(labels[labels.index == label_change.index[i]])==1 or int(labels[labels.index == label_change.index[i]])==2

                                    else
                                        dict(
                                            type="rect",
                                            # x-reference is assigned to the x-values
                                            xref="x",
                                            # y-reference is assigned to the plot paper [0,1]
                                            yref="paper",
                                            x0=label_change.index[i],
                                            y0=0.44,
                                            x1=label_change.index[i+1],
                                            y1=1,
                                            fillcolor="#E5ECF6", # default plotly background
                                            opacity=0.2,
                                            layer="below",
                                            line_width=0,
                                        )   
                                )

                            for i in range(len(label_change)-1)
    ]
    return background_color





