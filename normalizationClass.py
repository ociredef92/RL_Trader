class DataNormalization:

    def __init__(self, ts, roll, ob_levels, start):
        self.ts = ts
        self.roll = roll
        self.ob_levels = ob_levels
        self.ts_shape = self.ts.shape[1]
        self.roll_window = self.roll * self.ob_levels * self.ts_shape
        self.roll_step = self.ob_levels * self.ts_shape
        self.start = 0
        self.ts_stacked = self.get_ts_stack() # stack dataframe as default
        self.new_data = pd.Series([])
    
    def get_ts_stack(self):
        if self.ts_shape > 1:
            ts_stacked = self.ts.stack()
            print(ts_stacked)
            return ts_stacked

    def get_new_data(self):
        self.new_data = self.ts_stacked.iloc[(self.start+self.roll_window):(self.start+self.roll_window+self.roll_step)]
        self.start += self.roll_step
        return self.new_data

    def dyn_z(self):
        mean_rw = np.mean(self.ts_stacked.iloc[0:self.roll_window])
        std_rw = np.std(self.ts_stacked.iloc[0:self.roll_window])
        self.new_data = self.get_new_data()
        print(self.new_data)
        