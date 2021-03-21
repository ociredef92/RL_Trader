import pandas as pd
import numpy as np

import keras
import tensorflow as tf

from keras.models import load_model, Model, Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Input, Reshape, Conv2D, MaxPooling2D, Conv1D, LSTM #CuDNNLSTM
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.preprocessing.sequence import TimeseriesGenerator

import datetime
#tf.enable_eager_execution()


# set random seeds
np.random.seed(1)

# # BTC_ETH form 4 April till 1 Sept
# cache_file = '/home/pawel/Documents/LOB-data/cache/BTC_ETH/data-cache-1m.csv'

# cached_data = pd.read_csv(cache_file, index_col=0)

dt = datetime.datetime.strptime('01/01/20', '%d/%m/%y')
delta = datetime.timedelta(minutes=1)
l = []
#print(dt + delta)
for i in range(400):
    for j in range(10):
        l.append([dt, j, 1 + i*0.1 + j*0.01 , 2 + i*0.1 + j*0.01, 3 + i*0.1 + j*0.01, 4 + i*0.1 + j*0.01])
    dt += delta

cached_data = pd.DataFrame(l, columns=['Datetime', 'Level', 'Ask_Price', 'Ask_Size', 'Bid_Price', 'Bid_Size'])

# # convert to datetime format
cached_data['Datetime'] =  pd.to_datetime(cached_data['Datetime'], format='%Y-%m-%d %H:%M:%S')
# set index
#cached_data = cached_data.set_index(['Datetime', 'Level'])
print(cached_data.shape) # print shape flattened ob depth

"""## Resample (to test smaller model) and perform train test split
#### Train test split prior to normalization. 2 datasets normalized independently
"""

# # Resample data
# resample_freq = '5min'
# cached_data_res = cached_data.groupby([pd.Grouper(key='Datetime', freq=resample_freq), pd.Grouper(key='Level')]).last().reset_index()
cached_data_res = cached_data

lob_depth = 10 # 10 level of order book
train_test_split = int((cached_data_res.shape[0] / lob_depth) * 0.7) # slice reference for train and test
train_timestamps = cached_data_res['Datetime'].unique()[:train_test_split]
test_timestamps = cached_data_res['Datetime'].unique()[train_test_split:]

train_cached_data = cached_data_res[cached_data_res['Datetime'].isin(train_timestamps)].set_index(['Datetime', 'Level'])
test_cached_data = cached_data_res[cached_data_res['Datetime'].isin(test_timestamps)].set_index(['Datetime', 'Level'])

print(train_cached_data.head(100))
print(test_cached_data.head(100))
print(f'Train dataset shape: {train_cached_data.shape} - Test dataset shape: {test_cached_data.shape}')

"""##Import func_tools"""

# funct_tools pasted from commit 1902046485b8bad753b2195d55c205d19c81890e

def normalize(ts, norm_type='z_score', roll=0):


    return ts # debugging

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

    # assign labels based on alpha threshold
    labels = direction.apply(lambda x: 1 if x>alpha else (-1 if x<-alpha else 0))

    return labels


def get_pnl(px_ts, labels, trading_fee, long_only=True):
    '''
    Function to get pnl from a price time series and respective labels
    px_ts and labels series must be mergeable by index

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
        df['return'] = df['px'].pct_change()

        return ((df['labels'] * df['return']) + 1).cumprod() - 1 # plus one here instead



def cnn_data_reshaping(X, Y, T, conv_type='1D'):
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

    if conv_type == '1D':
        dataX = dataX.reshape(dataX.shape)# + (1,)) # no need to add the extra dimension for 1d conv

    elif conv_type == '2D':
        dataX = dataX.reshape(dataX.shape + (1,))

    #print(f'shape X:{dataX.shape}, shape Y:{dataY.shape}')

    return dataX, dataY


"""## Normalize - check how train and test distributions differ"""

# use normalize to calculate standardized z score version of the train data
train_z_prices = normalize(train_cached_data[['Ask_Price', 'Bid_Price']], norm_type='z_score', roll=0) # get norm prices
train_z_volumes = normalize(train_cached_data[['Ask_Size', 'Bid_Size']], norm_type='z_score', roll=0) # get norm volumes
train_z_df = pd.concat([train_z_prices, train_z_volumes], axis=1).reset_index() # concat along row index


# use normalize to calculate standardized z score version of the test data
test_z_prices = normalize(test_cached_data[['Ask_Price', 'Bid_Price']], norm_type='z_score', roll=0) # get norm prices
test_z_volumes = normalize(test_cached_data[['Ask_Size', 'Bid_Size']], norm_type='z_score', roll=0) # get norm volumes
test_z_df = pd.concat([test_z_prices, test_z_volumes], axis=1).reset_index() # concat along row index


"""## Reshape"""

# Get data in the desired format - similar to deep lob
# Steps: pivot data, transpose and reset index (easier to sort than columns),
# sort by order book level and then event type in order to have at each level
# ask_price, ask_size, bid_price, bid_size. Finally reset the index back and transpose

def reshape_lob_levels(z_df, output_type='array'):
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

train_depth, train_dt_index = reshape_lob_levels(train_z_df, output_type='array')
test_depth, test_dt_index = reshape_lob_levels(test_z_df, output_type='array')

# # visual check of df if needed
# df_depth , _= reshape_lob_levels(train_z_df, output_type='dataframe')
# df_depth

"""## Label"""

# to do: label train data and try training simple model

# best bid offer dataframe - from cached data
bbo_df = cached_data_res.reset_index()[cached_data_res.reset_index()['Level'] == 0]
bbo_df['Mid_Price'] = (bbo_df['Ask_Price'] + bbo_df['Bid_Price']) / 2

bbo_df.shape

# get labels on non-standardized mid prices
k = 50
alpha = 0.005
labels = get_labels(bbo_df['Mid_Price'], k, alpha)

#labels.plot()



"""## 1D CNN data: train_X & train_cat_Y"""

from keras.utils import np_utils

T = 100
labels_reshaped = labels.values.reshape(len(labels),1)

# Augment data to prepare for 1D convolution (add extra dimension back for 2D)

train_X, train_Y = cnn_data_reshaping(train_depth, labels_reshaped[:train_test_split], T, conv_type='2D')
test_X, test_Y = cnn_data_reshaping(test_depth, labels_reshaped[train_test_split:], T, conv_type='2D')

# Transform label into one hot encoded categories

# categorical labels - whole series to make sure train and test get the same encoding (correct?)
cat_labels = np_utils.to_categorical(np.array(labels_reshaped),3)

train_cat_Y = cat_labels[:train_test_split][T - 1:]
test_cat_Y = cat_labels[train_test_split:][T - 1:]
print(f'Encoded labels shape: train: {train_cat_Y.shape} - test: {test_cat_Y.shape}')

"""## Simple 1D Conv Model"""

# def simple_1d_CNN(verbose, epochs, batch_size):
#     #verbose, epochs, batch_size = 1, 5, 32
#     n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_cat_Y.shape[1]

#     #input_pm = Input(shape=(n_timesteps,n_features))

#     model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=1, input_shape=(n_timesteps,n_features)))
#     model.add(keras.layers.LeakyReLU(alpha=0.01))#(convsecond_2)

#     model.add(Conv1D(filters=64, kernel_size=1))
#     model.add(keras.layers.LeakyReLU(alpha=0.01))

#     model.add(Dropout(0.5))
#     #model.add(MaxPooling1D(pool_size=2))
#     model.add(Flatten())
#     #model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # fit network

#     return model

# # simple_model = simple_1d_CNN(1, 5, 32)
# # simple_model.fit(train_X, train_cat_Y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(test_X, test_cat_Y))

"""## Deep LOB Model"""

def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))
    
    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)
    
    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)

    # use the MC dropout here
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

deeplob = create_deeplob(100, 40, 64)

deeplob.summary()


#x = train_depth.reshape(train_depth.shape + (1,))
x = train_depth

y = np_utils.to_categorical(np.array(labels_reshaped),3)


generator = TimeseriesGenerator(
    x,
    y[:train_test_split],
    100,
    batch_size=64,
    shuffle=False,
    reverse=False
)

history = deeplob.fit(generator, epochs=20, verbose=1, validation_data=(test_X, test_cat_Y))

#history = deeplob.fit(train_X, train_cat_Y, epochs=20, batch_size=64, verbose=1, shuffle=False, validation_data=(test_X, test_cat_Y))



"""## Predictions"""

predictions = deeplob.predict(test_X, verbose=1)
#predictions = model.predict(test_X, verbose=1) #to do argmamx on top of this
# np.argmax(predictions, axis=1)
# # not matching correctly to the original categories
# categories = [-1, 0, 1]
# predicted_labels = [categories[i] for i in predictions]

# import collections
# max_index = np.argmax(predictions, axis=1)
# collections.Counter(max_index)

"""## Plotting results"""

# # Plot predictions
# fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])

# #fig = px.line(title='<b>Data values and labels</b>')#, name='sin wave'
# fig.update_layout(title='<b>Visual check: model predictions</b>', title_x=0.5)

# fig.add_trace(go.Scatter(y=bbo_df['Mid_Price'], x=bbo_df['Datetime'], name='currency pair'))
# fig.add_trace(go.Scatter(y=labels, x=bbo_df['Datetime'], name='labels'),row=1, col=1, secondary_y=True)

# fig.add_trace(go.Scatter(y=predicted_labels, x=bbo_df['Datetime'][train_test_split:], name='predictions'),
#                           row=1, col=1, secondary_y=True)

# fig.show()

# to do fix prediction labels
# better plotting: need a  clearer way to display predictions and original labels
# add model results in dash app
