import time

# from requests import get
# from json import loads

# for i in range(22):
#     start = time.time()
#     response = get('https://poloniex.com/public?command=returnOrderBook&currencyPair=USDT_BTC&depth=10')
#     book = loads(response.text)

#     print(f'{time.time() - start}s')

#     # 0.2612192630767822s
#     # 0.35961484909057617s
#     # 0.41544365882873535s
#     # 0.3315165042877197s
#     # 0.35619521141052246s
#     # 0.44608044624328613s
#     # 0.37185025215148926s
#     # 0.4041721820831299s
#     # 0.33335447311401367s
#     # 0.3324868679046631s
#     # 0.34612536430358887s
#     # 0.3390181064605713s
#     # 0.2800741195678711s
#     # 0.2975199222564697s
#     # 0.30638980865478516s
#     # 0.3447885513305664s
#     # 0.37688660621643066s
#     # 0.32017087936401367s
#     # 0.2861654758453369s
#     # 0.33791422843933105s
#     # 0.3473207950592041s
#     # 0.44145846366882324s


import pandas as pd
import numpy as np
import sqlite3


table_name = 'USDT_BTC'

conn = sqlite3.connect('/home/pawel/lob_sqlite.db')

# TODO - explore storing levels as columns 

create_table_sql = """ CREATE TABLE IF NOT EXISTS USDT_BTC (
                                    id integer PRIMARY KEY,
                                    timestamp integer,
                                    Level integer,
                                    Ask_Price float,
                                    Ask_Size float,
                                    Bid_Price float,
                                    Bid_Size float
                                ); """

c = conn.cursor()
c.execute(create_table_sql)    


# data = pd.read_csv('/home/pawel/Documents/LOB-data/cache/USDT_BTC/USDT_BTC--10seconds--2020_04_03--2020_12_24.csv.gz', compression='gzip', parse_dates=['Datetime'])
# data = data.tail(432000) # 7200*6*10

# data['timestamp'] = data['Datetime'].astype(np.int64) // 10**9
# data.drop('Datetime', axis=1, inplace=True)

# data.drop('Unnamed: 0', axis=1, inplace=True)
# data.drop('Unnamed: 0.1', axis=1, inplace=True)
# data.drop('Sequence', axis=1, inplace=True)

# data.to_sql(table_name, conn, if_exists='append', index=False)

start = time.time()

data = pd.read_sql_query(f'SELECT * from {table_name} WHERE timestamp > 1608901190', conn)

print(f'{time.time() - start}s')

conn.close()

# read 432000 rows - 1.087900161743164s
# read 10 rows - 0.030169010162353516s