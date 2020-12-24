import numpy as np
import pandas as pd

class DataNormalization:

    def __init__(self, ts, roll, ob_levels, start=0):
        ''' 
            ts: pd.Series or pd.Dataframe. If dataframe, need to have cols that can be normalized
                together, like all prices or sizes

            roll: int, rolling window (depends on frequency of data passed)

            ob_levels: int, orderbook depth. Assumed to be constant throughtout all timeseries
            
            start: int, at which point of the timeseries the rolling start. Has to be a multiple of 
                    ob_levels * n df columns
        '''
        self.ts = ts
        self.roll = roll
        self.ob_levels = ob_levels
        self.ts_shape = self.ts.shape[1]
        self.roll_window = self.roll * self.ob_levels * self.ts_shape
        self.roll_step = self.ob_levels * self.ts_shape
        self.start = start
        self.ts_stacked = self.get_ts_stack() # stack dataframe as default
        self.new_data = pd.Series()
        self.dyn_ts = pd.Series()
    
    def get_ts_stack(self):
        ''' Flatten dataframe into a series if more than 1 column is passed '''
        if self.ts_shape > 1:
            self.ts_stacked = self.ts.stack()
            #print(self.ts_stacked)
        else:
            self.ts_stacked = self.ts
        return self.ts_stacked

    def get_new_data(self):
        ''' Add 1 roll step to the self.start variable, to get next timestep from dataframe '''
        self.start += self.roll_step
        self.new_data = self.ts_stacked.iloc[(self.start+self.roll_window):(self.start+self.roll_window+self.roll_step)]
        return self.new_data

    def get_one_dyn_z(self):
        ''' Calculate 1 period dynamic z score - 1/100th of a second'''
        mean_rw = np.mean(self.ts_stacked.iloc[self.start:self.roll_window+self.start])
        std_rw = np.std(self.ts_stacked.iloc[self.start:self.roll_window+self.start])
        # self.start is updated in get_new_data, so get_new_data() has to be executed after mean_rw and std_rw
        self.new_data = self.get_new_data() 
        #print(self.ts_stacked.iloc[self.start:self.roll_window+self.start])
        #print(self.new_data)
        z_rw = (self.new_data - mean_rw) / std_rw
        return z_rw
        
    def get_ts_dyn_z(self):
        ''' Loop through all time series - much slower than pandas rolling implementation '''
        while self.roll_window+self.start <= self.ts_stacked.shape[0]:
            self.dyn_ts = pd.concat([self.dyn_ts, self.get_one_dyn_z()])
        return self.dyn_ts
