import pandas as pd
import time
import calendar
from datetime import datetime
from requests import get
from json import loads
import sys, os

start_date = '2021-01-01'
end_date = '2021-03-01'

response = get(f'https://poloniex.com/public?command=returnTicker')
if response.status_code == 200:
    pairs = loads(response.text)
else:
    raise Exception(response.content)

for pair in pairs.keys():

    start = calendar.timegm(datetime.strptime(start_date, "%Y-%m-%d").timetuple())
    end = calendar.timegm(datetime.strptime(end_date, "%Y-%m-%d").timetuple())

    trades_folder = os.path.realpath(os.path.join(os.getcwd(), f'trades/{pair}'))

    if not os.path.exists(trades_folder):
        os.makedirs(trades_folder)

    interval = 60 * 60 * 24 # 24 hours default

    if pair in ['BTC_XMR', 'BTC_LTC', 'BTC_DOGE', 'BTC_DASH', 'USDT_XFLR', 'USDT_FARM', 'USDT_TRX', 'USDT_ETH', 'BTC_ETH']:
        interval = 60 * 3 # 3 minutes
    elif pair in ['USDT_BTC']:
        interval = 60 * 1 # 1 minute

    for day in range(start, end, 60*60*24): #loop through daysstr
        trades_day = []
        trades_day_file_name = os.path.join(trades_folder, f"{pair}-{datetime.utcfromtimestamp(day).strftime('%Y%m%d')}.csv.gz")
        if not os.path.exists(trades_day_file_name):

            for minute in range(day, day + 60*60*24, interval):# assuming no more than 1000 trades (poliniex api restriction) in 1 interval window
                response = get(f'https://poloniex.com/public?command=returnTradeHistory&currencyPair={pair}&start={minute}&end={minute+interval-1}')
                if response.status_code == 200:
                    trades = loads(response.text)
                    trades_day += trades
                    print(datetime.utcfromtimestamp(minute).strftime(f'Fetched {len(trades)} trades for {pair} %Y-%m-%d  %H:%M:%S'))
                    if len(trades) == 1000:
                        raise Exception(f'1000 trades')

                else:
                    raise Exception(response.content)
            df = pd.DataFrame(trades_day)
            if not trades_day:
                print(f'No trades in {trades_day_file_name}')
            else:
                df.sort_values(by=['date'])
            df.to_csv(trades_day_file_name, index=False, compression='gzip')
        else:
            print(f'Found {trades_day_file_name}')
