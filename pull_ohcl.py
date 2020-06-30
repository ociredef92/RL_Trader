import ccxt
import pandas as pd
import time, datetime
import json
import sys, os

exchange = ccxt.poloniex({
    'apiKey': '',
    'secret': '',
})

symbol = 'BTC/USDT'
start = int(exchange.parse8601('2020-04-03 00:00:00') / 1000)
end = int(exchange.parse8601('2020-06-29 00:00:00') / 1000)

data = []

cache_folder = os.path.realpath(os.path.join(os.getcwd(), 'cache'))

if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

for i in range(start, end, 60):

    cache_file_name = os.path.join(cache_folder, symbol.replace('/', '-') + str(i) + '.json')
    if not os.path.exists(cache_file_name):

        print(datetime.datetime.utcfromtimestamp(i).strftime('Fetching trades for %Y-%m-%d %H:%M:%S'))
        # assuming no more than 1000 trades (poliniex api restriction) in a single minute
        trades = exchange.fetch_trades(symbol, i * 1000, None, params={"end": i + 59})
        with open(cache_file_name, 'w') as cachefile:
            json.dump(trades, cachefile)

    else:
        print(datetime.datetime.utcfromtimestamp(i).strftime('Getting trades from cache for %Y-%m-%d %H:%M:%S'))        
        with open(cache_file_name, 'r') as cachefile:
            trades = json.loads(cachefile.read())
    data += trades
    #time.sleep(1)

candles = exchange.build_ohlcv(data, timeframe='10s')

header = ['t', 'o', 'h', 'l', 'c', 'v']
df = pd.DataFrame(candles, columns=header)
df.to_csv('btc_usdt.csv', index=False)