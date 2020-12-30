from apscheduler.schedulers.background import BackgroundScheduler
from requests import get
from json import loads
from flask import Flask

def get_book():
    response = get('https://poloniex.com/public?command=returnOrderBook&currencyPair=USDT_BTC&depth=10')
    book = loads(response.text)
    print(book)

sched = BackgroundScheduler(daemon=True)
sched.add_job(get_book, 'cron', second='*')
sched.start()

app = Flask(__name__)
if __name__ == "__main__":
    app.run('0.0.0.0',port=5000)