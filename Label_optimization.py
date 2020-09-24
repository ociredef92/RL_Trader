#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
import multiprocessing
import func_tools as ft


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# BTC_ETH form 4 April till 1 Sept - downloaded locally
#cache_file = 'https://aws-sam-cli-managed-default-samclisourcebucket-1fti72g7fv2bu.s3.eu-west-2.amazonaws.com/data-cache-1m.csv'

cached_data = pd.read_csv('/Users/federicotampieri/Downloads/RL_Trader_Caching/BTC_ETH_1m/data-cache-1m.csv', 
                          index_col=0)

# # convert to datetime format
cached_data['Datetime'] =  pd.to_datetime(cached_data['Datetime'], format='%Y-%m-%d %H:%M:%S')
# set index
#cached_data = cached_data.set_index(['Datetime', 'Level'])
print(cached_data.shape) # print shape flattened ob depth
cached_data.head(5)


# In[5]:


# Further resampling?
resample_freq = '1min'
cached_data_res = cached_data.groupby([pd.Grouper(key='Datetime', freq=resample_freq), pd.Grouper(key='Level')]).last().reset_index()

# get data. Run label opt on dyn z score
# best bid offer dataframe - from cached data
bbo_df = cached_data_res.reset_index()[cached_data_res.reset_index()['Level'] == 0]
bbo_df['Mid_Price'] = (bbo_df['Ask_Price'] + bbo_df['Bid_Price']) / 2
print(bbo_df.shape)
bbo_df.head()


# In[6]:


# rudimental grid search
k_plus_range = np.arange(10,50,1)
k_minus_range = np.arange(10,50,1)
alpha_range = np.arange(0.001,0.05, 0.001)
long_only = False
trading_fee = 0.000712

px_ts = bbo_df.set_index('Datetime')['Mid_Price']
len_px_ts = px_ts.shape[0]


# In[7]:


#%timeit get_labels(px_ts, 30, 0.01)
# %timeit ft.get_labels(px_ts, 30, 0.01)
# labels = ft.get_labels(px_ts, 30, 0.01)
# %timeit ft.get_pnl(px_ts, labels, trading_fee, long_only=long_only)
# %timeit get_pnl(px_ts, labels, trading_fee)


# In[8]:


# # Label grid search on single process
# print(f'Number of iterations: {k_plus_range.shape[0] * k_minus_range.shape[0] * alpha_range.shape[0]}')
# def label_grid_search(alpha_range, k_plus_range, k_minus_range):

#     gs_list = [] # empty list to store values
#     for k_plus in k_plus_range:
#         for k_minus in k_minus_range:
#             for alpha in alpha_range:
            

#                 labels = ft.get_labels(px_ts, k_plus=k_plus, k_minus=k_minus, alpha=alpha, long_only=long_only)
#                 pnl, idx = ft.get_pnl(px_ts, labels, trading_fee)

#                 # get for how long labels are "in the market"
#                 unique_labels, counts_labels = np.unique(labels.values, return_counts=True)
#                 percent_labels = counts_labels / counts_labels.sum()

#     #             a_ext = np.concatenate(( [0], labels.values, [0])) # extend array for comparison
#     #             idx = np.flatnonzero(a_ext[1:] != a_ext[:-1]) # non zero indices
#     #             idx.shape[0] # how many transactions

#                 # show with how many trades
#                 unique_trades, counts_trades = np.unique(labels.values[[idx]], return_counts=True)

#                 try:
#                     gs_list.append([k_plus, k_minus, alpha, pnl.values[-1], pnl.shape[0], 
#                                   percent_labels[0], percent_labels[1], percent_labels[2],
#                                   counts_trades[0], counts_trades[1], counts_trades[2]])

#                 except:
#                     gs_list.append([k, alpha, 0, len_px_ts, 0, 0, 0, 0, 0, 0]) # if parameters generate no labels, empty series is returned

#     return gs_list


# In[9]:


# # Time to execute on single
# start_time = time.time()
# gs_list_single = label_grid_search(alpha_range, k_plus_range, k_minus_range)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[10]:


# pnl_df_single = pd.DataFrame(gs_list_single, columns=['k_plus', 'k_minus', 'alpha', 'pnl', 'in_mkt', 
#                                         'pct_long', 'pct_hold', 'pct_short',
#                                         'trades_buy', 'trade_hold', 'trades_sell']).sort_values(by='pnl', ascending=False)

# pnl_df_single


# In[11]:


# Label grid search on multi processes
print(f'''k_plus space size: {k_plus_range.shape[0]}  
k_minus space size: {k_minus_range.shape[0]}
alpha space size: {alpha_range.shape[0]}''')
print(f'Number of iterations: {k_plus_range.shape[0] * k_minus_range.shape[0] * alpha_range.shape[0]}')
def label_grid_search_multiprocessing(alpha, k_plus, k_minus):
    gs_list_multi = []
    #global gs_list

    labels = ft.get_labels(px_ts, k_plus, k_minus, alpha, long_only=long_only)
    pnl, idx = ft.get_pnl(px_ts, labels, trading_fee)

    # get for how long labels are "in the market"
    unique_labels, counts_labels = np.unique(labels.values, return_counts=True)
    percent_labels = counts_labels / counts_labels.sum()

    #a_ext = np.concatenate(( [0], labels.values, [0])) # extend array for comparison
    #idx = np.flatnonzero(a_ext[1:] != a_ext[:-1]) # non zero indices
    #idx.shape[0] # how many transactions

    # show with how many trades
    unique_trades, counts_trades = np.unique(labels.values[[idx]], return_counts=True)

    try:
        return [k_plus, k_minus, alpha, pnl.values[-1], pnl.shape[0], 
              percent_labels[0], percent_labels[1], percent_labels[2],
              counts_trades[0], counts_trades[1], counts_trades[2]]

    except:

        return[k_plus, k_minus, alpha, 0, len_px_ts, 0, 0, 0, 0, 0, 0] # if parameters generate no labels, empty series is returned

    return gs_list_multi


# In[12]:


# Time to execute on multi
number_of_workers = 2
#gs_list = multiprocessing.Manager().list()
inputs_tuple = tuple([(a, k_plus, k_minus) 
                      for a in alpha_range 
                      for k_plus in k_plus_range 
                      for k_minus in k_minus_range])

start_time = time.time()
with multiprocessing.Pool(number_of_workers) as p:
    res = p.starmap(label_grid_search_multiprocessing, inputs_tuple)

    p.close()
    p.join()
print("--- %s seconds ---" % (time.time() - start_time))


# In[13]:


#gs_list = label_grid_search()
pnl_df_multi = pd.DataFrame(res, columns=['k_plus', 'k_minus','alpha', 'pnl', 'in_mkt', 
                                        'pct_long', 'pct_hold', 'pct_short',
                                        'trades_buy', 'trade_hold', 'trades_sell']).sort_values(by='pnl', ascending=False)

pnl_df_multi


# In[14]:


# transfer grid search result to csv
pnl_df_multi.to_csv('/Users/federicotampieri/Downloads/RL_Trader_Caching/BTC_ETH_1m/label_grid_search.csv')


# In[15]:


#pd.read_csv('/Users/federicotampieri/Downloads/RL_Trader_Caching/BTC_ETH_1m/label_grid_search.csv')


# In[ ]:




