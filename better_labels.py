import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from func_tools import normalize, get_labels, cnn_data_reshaping, reshape_lob_levels, plot_labels, label_insights, get_pnl
import time
import plotly_express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# %%
model_name = 'dynz_score_lob_v3_10s.h5'
security = 'USDT_BTC'
root_caching_folder = "/home/pawel/Documents/LOB-data/cache"
#frequency = timedelta(seconds=10)
norm_type = 'dyn_z_score'
trading_fee=0.000712

# labelling inputs
k_plus = 30#60
k_minus = 30#60
alpha = 0.001#0.0005
roll = 7200 * 6 # step from minute to 10 second data


# %%
data = pd.read_csv(f'{root_caching_folder}/{security}/USDT_BTC--10seconds--sample2000000.csv.gz', index_col=0, compression='gzip')
lob_depth = data['Level'].max() + 1

# Train test split
train_test_split = int((data.shape[0] / lob_depth) * 0.7) # slice reference for train and test

train_timestamps = data['Datetime'].unique()[:train_test_split] # TODO unique is redundant after filling gaps during peprocessing
test_timestamps = data['Datetime'].unique()[train_test_split:]

train_cached_data = data[data['Datetime'].isin(train_timestamps)].set_index(['Datetime', 'Level'])
test_cached_data = data[data['Datetime'].isin(test_timestamps)].set_index(['Datetime', 'Level'])

print(f'Train dataset shape: {train_cached_data.shape} - Test dataset shape: {test_cached_data.shape}')
    


# Parallelized data size & price standardization for train and test set
number_of_workers = 4
inputs = (
    (train_cached_data[['Ask_Price', 'Bid_Price']], lob_depth, 'dyn_z_score', roll),
    (train_cached_data[['Ask_Size', 'Bid_Size']], lob_depth, 'dyn_z_score', roll),
    (test_cached_data[['Ask_Price', 'Bid_Price']], lob_depth, 'dyn_z_score', roll),
    (test_cached_data[['Ask_Size', 'Bid_Size']], lob_depth, 'dyn_z_score', roll)
    )

start_time = time.time()

with multiprocessing.Pool(number_of_workers) as p:
    res = p.starmap(normalize, inputs)
    res = list(res)
    #print(res)
    p.close()   
    p.join()
    
train_dyn_prices, train_dyn_volumes, test_dyn_prices, test_dyn_volumes = res[0], res[1], res[2], res[3]
print("--- %s seconds ---" % (time.time() - start_time))

# train_dyn_prices = normalize(train_cached_data[['Ask_Price', 'Bid_Price']], lob_depth, 'dyn_z_score', roll)
# train_dyn_volumes = normalize(train_cached_data[['Ask_Size', 'Bid_Size']], lob_depth, 'dyn_z_score', roll)
# test_dyn_prices =normalize(test_cached_data[['Ask_Price', 'Bid_Price']], lob_depth, 'dyn_z_score', roll)
# test_dyn_volumes = normalize(test_cached_data[['Ask_Size', 'Bid_Size']], lob_depth, 'dyn_z_score', roll)

# concat prices and volumes back together and top level (useful for Dash)
train_dyn_df = pd.concat([train_dyn_prices, train_dyn_volumes], axis=1).reset_index() # concat along row index
#train_dyn_df[train_dyn_df['Level']==0].to_csv(f'{root_caching_folder}/{security}/TRAIN-{lob_depth}-{norm_type}-{roll}.csv') # save top level to csv 

test_dyn_df = pd.concat([test_dyn_prices, test_dyn_volumes], axis=1).reset_index() # concat along row index
#test_dyn_df[test_dyn_df['Level']==0].to_csv(f'{root_caching_folder}/{security}/TEST-{lob_depth}-{norm_type}-{roll}.csv') # save top level to csv 

#display(train_dyn_df.describe()) # check train data overview
#display(test_dyn_df.describe()) # check test data overview

# 1 reshape to a format suitable for training
# 2 get mid px from normalized data
# 3 get labels from norm mid prices
# 4 labels one hot encoding

# train
train_depth_dyn, train_dt_index_dyn = reshape_lob_levels(train_dyn_df, output_type='array') # 1 train dataset
mid_px_train_dyn = pd.Series((train_depth_dyn[:,2] + train_depth_dyn[:,0]) / 2) # 2

mid_px_train_real = train_cached_data[train_cached_data.index.isin([0], level='Level')].loc[: , ['Ask_Price', 'Bid_Price']].mean(axis=1)

labels_dyn_train = get_labels(mid_px_train_dyn, mid_px_train_real, k_plus, k_minus, alpha, long_only=False) # 3
encoded_train_labels = np_utils.to_categorical(labels_dyn_train.values,3) # 4 train labels

# test
test_depth_dyn, test_dt_index_dyn = reshape_lob_levels(test_dyn_df, output_type='array') # 1 test dataset
mid_px_test_dyn = pd.Series((test_depth_dyn[:,2] + test_depth_dyn[:,0]) / 2) # 2

mid_px_test_real = test_cached_data[test_cached_data.index.isin([0], level='Level')].loc[: , ['Ask_Price', 'Bid_Price']].mean(axis=1)

labels_dyn_test = get_labels(mid_px_test_dyn, mid_px_test_real, k_plus, k_minus, alpha, long_only=False) # 3
encoded_test_labels = np_utils.to_categorical(labels_dyn_test.values,3) # 4 test labels


# Information about the newly generated labels
print('Train Labels')
train_transact_dyn = label_insights(labels_dyn_train)
print('\nTest Labels')
test_transact_dyn = label_insights(labels_dyn_test)
print(f'\nLabels Train as pctg of total: {test_transact_dyn/(test_transact_dyn+train_transact_dyn)}')


# Labels sanity check
def plot_data(norm_mid_px, labels, start, end, train_test_switch, y0=0):

    fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])
    fig.update_layout(title=f'<b>Visual check: {train_test_switch}</b>', title_x=0.5)

    fig.add_trace(go.Scatter(y=norm_mid_px.values[start:end], x=norm_mid_px.index[start:end], name='mix_px_dyn_train'))   
    #fig.add_trace(go.Scatter(y=labels_dyn_train[start:end], name='labels_encoded'), secondary_y=True)

    background_color = plot_labels(labels[start:end], y0) # funct_tools formula to plot labels
    
    fig.update_layout(width=1200, 
        height=600,
        shapes=background_color,
        xaxis2= {'anchor': 'x','overlaying': 'x', 'side': 'top'},
        yaxis_domain=[0, 1])

    return fig


# plot train data
plot_data(mid_px_train_dyn, labels_dyn_train, 0, 30000, train_test_switch='train',y0=0)

# plot test data
plot_data(mid_px_test_dyn, labels_dyn_test, 0, 10000, train_test_switch='test',y0=0)

