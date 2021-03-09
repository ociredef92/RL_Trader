import numpy as np
import pandas as pd
import plotly_express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Labels
def plot_labels_line(px_ts, labels, title='Labels', **kwargs):
    '''Plot labels against price.
    Takes two pandas timeseries as inputs. These need to be subsets of the same
    DataFrame or have same length
    '''
    #print(kwargs)
    # check index
    condition = (px_ts.index == labels.index).sum()
    assert condition == px_ts.shape[0] == labels.shape[0], 'px_ts and labels must have the same index to be correctly plotted'

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=px_ts, x=px_ts.index, name='Price'), secondary_y=False)
    fig.add_trace(go.Scatter(y=labels, x=labels.index, name='Labels', marker=dict(color='rgba(240, 52, 52, 0.3)')), 
        secondary_y=True)


    for arg, key in zip(kwargs.values(), kwargs.keys()):
        if 'label' in key or 'direction' in key:
            fig.add_trace(go.Scatter(y=arg, x=arg.index, name=key), secondary_y=True)
        else:
            fig.add_trace(go.Scatter(y=arg, x=arg.index, name=key), secondary_y=False)

    fig.update_layout(title=f'<b>{title}</b>', width=1200, height=800)
    fig.update_yaxes(title_text='ccy', fixedrange= False, secondary_y=False)
    fig.update_yaxes(title_text='label', secondary_y=True)
    #fig.update_yaxes(fixedrange= False)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
        )
    )

    return fig.show()


# Labels insights
def plot_trades_distribution(df_trades, bin_size=0.0001, metric='gross_returns', fig_width=900, fig_height=550):
    ''' Plot trades distribution and approx distribution curve.
    Takes as an input df_trades from stratgy pnl, bin_size (default 1bp) and gross return as metrix
    '''

    trades = [df_trades['gross_returns'].values]
    labels = ['Trades']
    fig = ff.create_distplot(trades, labels, bin_size = 0.0001, show_rug=False)
    # Add shapes
    avg = np.mean(trades)
    stdev = np.std(trades)

    fig.add_shape(type="line", yref='paper',
        x0=avg, y0=0, x1=avg, y1=1,
        line=dict(color="RoyalBlue",width=2)
    )

    fig.add_shape(type="line", yref='paper',
        x0=avg+stdev, y0=0, x1=avg+stdev, y1=1,
        line=dict(color="RoyalBlue",width=2, dash="dot")
    )

    fig.add_shape(type="line", yref='paper',
        x0=avg-stdev, y0=0, x1=avg-stdev, y1=1,
        line=dict(color="RoyalBlue",width=2, dash="dot")
    )

    fig.add_shape(type="line", yref='paper',
        x0=0, y0=0, x1=0, y1=1,
        line=dict(color="rgba(0, 0, 0, 0.5)",width=2, dash="dashdot")
    )

    fig.update_layout(title=f"<b>Trades distribution - {metric}</b>", width=fig_width, height=fig_height, xaxis=dict(tickformat=',.3%'))
    fig.show()    


def plot_trades_length_overview(df_trades, x='trade_len',  y='gross_returns'):
    ''' Plot visual insight for lavels on x variable (default "trade_len"):
    1) histogram with count of x
    2) histogram with x vs average y (default "gross_returns")
    3) individual trades x vs y

    Takes as an input df_trades from stratgy pnl, with x and y being columns of df_trades
    '''

    max_trade_length = int(df_trades['trade_len'].max())
    hist_trade_length = px.histogram(df_trades, x=x, color='labels', title=f'<b>{x}</b>')
    avg = df_trades['trade_len'].mean() # average trade length
    hist_trade_length.add_shape(type="line", yref='paper',
        x0=avg, y0=0, x1=avg, y1=1,
        line=dict(color="rgba(0, 0, 0, 0.5)",width=2, dash="dashdot")
    )
    hist_trade_length.show()

    # Plot net returns (by length and average returns)
    hist_ret_len = px.histogram(df_trades, x=x, y=y, histfunc='avg', color='labels', nbins=max_trade_length, title=f'<b>{y} by {x}</b>')
    hist_ret_len.update_layout(yaxis=dict(tickformat=',.3%'))
    hist_ret_len.show()

    # Plot individual trades vs trade length
    avg_net_by_length = df_trades.groupby('trade_len')['gross_returns'].mean()
    ret_len_scatter = px.scatter(df_trades, x=x, y=y, color=df_trades['labels'].astype('str'), opacity=0.3, title=f'<b>{y} single trades</b>')
    ret_len_scatter.add_trace(go.Scatter(x=avg_net_by_length.index, y=avg_net_by_length.values, mode='lines', name='Average'))
    ret_len_scatter.update_layout(yaxis=dict(tickformat=',.3%'))
    ret_len_scatter.show()