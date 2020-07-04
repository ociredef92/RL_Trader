#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import json
import glob, os

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px


# In[2]:


directory = 'Downloads/USDT_BTC_2020/04/'
directory_list = sorted([x[0] for x in os.walk(directory)][1:])


# In[4]:


# Navigate through folder structure
list_quotes = []
number_files_processed = 0

for folder in directory_list[5:20]: # limit number of day to handle memory
    path = '{}/'.format(folder)
    
    for jsonfilename in glob.glob("{}*.gz".format(path)):

        # Read zipped files
        with gzip.GzipFile(jsonfilename, 'r') as fin:    # gzip
            json_bytes = fin.read()                      # bytes

        json_str = json_bytes.decode('utf-8')            # decode bytes
        data = json.loads(json_str)                      # to python object
        

        # Create list of lists for each timestamp
        for key in data.keys():
            list_quotes.append(list(zip(
                                    [i[0] for i in data.get(key).get('asks')], # ask px
                                    [i[1] for i in data.get(key).get('asks')], # ask size
                                    [i[0] for i in data.get(key).get('bids')], # bid px
                                    [i[1] for i in data.get(key).get('bids')], # bid size
                                    list(range(len(data.get(key).get('bids')))), # ob level - assuming same for both
                                    [data.get(key).get('seq')]*100, # seq
                                    [data.get(key).get('isFrozen')]*100, # frozen flag
                                    [datetime.strptime(key[-15:], '%Y%m%d_%H%M%S')]*100  #datetime 
                                ) ) ) 
        number_files_processed += 1
        
print(f'Amount of hours processed {number_files_processed}')


# In[5]:


df_quotes_all = pd.DataFrame([y for x in list_quotes for y in x], #flatten the list of lists structure
                          columns = ['Ask_Price', 'Ask_Size', 'Bid_Price', 'Bid_Size',
                                        'Level', 'Seq', 'isFrozen', 'Datetime']).sort_values(by = ['Datetime', 'Level'])
print(df_quotes_all.shape)
list_quotes = [] # free up memory


# In[6]:


# enforce data types
df_quotes_all['Ask_Price'] = df_quotes_all['Ask_Price'].astype('float64')
df_quotes_all['Bid_Price'] = df_quotes_all['Bid_Price'].astype('float64')

# create USD notional columns
df_quotes_all['Bid_Notional'] = df_quotes_all['Bid_Size'] * df_quotes_all['Bid_Price']
df_quotes_all['Ask_Notional'] = df_quotes_all['Ask_Size'] * df_quotes_all['Ask_Price']


# In[7]:


# First level orderbook (bbo - best bid offer)
df_quotes_bbo = df_quotes_all[df_quotes_all.Level == 0].copy()

# bbo spread and mid px
df_quotes_bbo['Mid_Price'] = (df_quotes_bbo['Ask_Price'] + df_quotes_bbo['Bid_Price'])/2
df_quotes_bbo['spread'] = (df_quotes_bbo['Ask_Price'] - df_quotes_bbo['Bid_Price'])/df_quotes_bbo['Mid_Price']

# bbo hourly bars
df_bbo_bars= df_quotes_bbo.groupby(pd.Grouper(key='Datetime', freq='1h')).agg({'Mid_Price':['mean', 'max', 'min', 
                                                                                            'first','last', 'count',
                                                                                            'std'],
                                                                               'spread': 'mean' }  ) 

# rename columns
df_bbo_bars.columns = ['mid_mean' , 'mid_high', 'mid_low', 'mid_open', 'mid_close', 'mid_#_obs', 
                       'mid_std', 'mean_spread']


# In[8]:


# Order Book depth
# spread thresholds - on each side of the orderbook
spread_threshold_tight = 0.0025
spread_threshold_medium = 0.0050
spread_threshold_wide = 0.0100


# In[9]:


# spread to bid - merge with bbo mid px, calculate respective spreads
bid_df = pd.merge(df_quotes_all[['Datetime', 'Bid_Price', 'Bid_Size', 'Bid_Notional']], 
                  df_quotes_bbo[['Mid_Price', 'spread','Datetime']], 
                  how = 'left', left_on = 'Datetime', right_on = 'Datetime')

bid_df['Bid_Spread'] = (bid_df['Mid_Price'] - bid_df['Bid_Price']) / bid_df['Mid_Price'] # check the math

# spread to ask - merge with bbo mid px, calculate respective spreads
ask_df = pd.merge(df_quotes_all[['Datetime', 'Ask_Price', 'Ask_Size', 'Ask_Notional']], 
                  df_quotes_bbo[['Mid_Price', 'spread','Datetime']], 
                  how = 'left', left_on = 'Datetime', right_on = 'Datetime')

ask_df['Ask_Spread'] = np.abs((ask_df['Mid_Price'] - ask_df['Ask_Price']) / ask_df['Mid_Price']) # check the math
                       

# Filter by spread threshold, only keeping tightest part of the order book
bid_tight_depth = bid_df[bid_df['Bid_Spread'] <= spread_threshold_tight].groupby('Datetime')['Bid_Notional'].sum()
bid_medium_depth = bid_df[bid_df['Bid_Spread'] <= spread_threshold_medium].groupby('Datetime')['Bid_Notional'].sum()
bid_wide_depth = bid_df[bid_df['Bid_Spread'] <= spread_threshold_wide].groupby('Datetime')['Bid_Notional'].sum()

ask_tight_depth = ask_df[ask_df['Ask_Spread'] <= spread_threshold_tight].groupby('Datetime')['Ask_Notional'].sum()
ask_medium_depth = ask_df[ask_df['Ask_Spread'] <= spread_threshold_medium].groupby('Datetime')['Ask_Notional'].sum()
ask_wide_depth = ask_df[ask_df['Ask_Spread'] <= spread_threshold_wide].groupby('Datetime')['Ask_Notional'].sum()


bid_ask_depth_df = pd.concat([bid_tight_depth, bid_medium_depth, bid_wide_depth,
                              ask_tight_depth, ask_medium_depth, ask_wide_depth], axis = 1)

bid_ask_depth_df.columns = ['bid_tight_depth', 'bid_medium_depth', 'bid_wide_depth',
                              'ask_tight_depth', 'ask_medium_depth', 'ask_wide_depth']
ba_depth_bars = bid_ask_depth_df.reset_index().groupby(pd.Grouper(key='Datetime', freq='1h')).mean()


# In[10]:


# Order book depth - hourly averages
hourly_df = pd.merge(df_bbo_bars, ba_depth_bars, how = 'left', left_index = True, right_index = True)


# In[11]:


# Plot the results
df_plot = hourly_df.reset_index()

# Plot px
fig_px = px.line(df_plot, x = 'Datetime',  y = 'mid_mean')
#fig_px.update_xaxes(rangeslider_visible=True)
fig_px.show()

# Plot Spread
fig_spread = px.line(df_plot, x = 'Datetime', y = 'mean_spread')
fig_spread.show()

# Plot Depth
df_plot = pd.melt(df_plot,id_vars=['Datetime'], value_vars=['bid_tight_depth', 'bid_medium_depth', 'bid_wide_depth',
                                                          'ask_tight_depth', 'ask_medium_depth', 'ask_wide_depth'])
fig_depth = px.line(df_plot, x='Datetime', y='value', color='variable')
fig_depth.show()

