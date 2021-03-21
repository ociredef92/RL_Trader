import numpy as np
import pandas as pd
import plotly_express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Px timeseries
def plot_timeseries(ts_list=[], primary_axis=[], legend=[], sample_size=360, width=900, height=500):
    '''Plot nth datapoints for the provided timeseries

    sample_size: integer. If input data is at 10s intervals, 30 would results in one datapoint
        plotted every 5 minutes, 360 every hour and 8640 every day
    ts_list, px_test: list of pandas timeseries with a datetime index
    primary_axis: list with boolean specifying whether a timeseries will be plotted on the primary axis or not
        If not specified, all timeseries will be plotted on primary axis
    legend: list with strings specifying legend labels. If not specified, legend will not be shown
    '''

    # if primary_axis or legend have values, they must have same len as ts_list
    if len(primary_axis)>0:
        assert len(ts_list) == len(primary_axis), "Specify to which axis each timeseries belongs"
    else:
        primary_axis = [False for ts in range(len(ts_list))]
    
    if len(legend)>0:
        assert len(ts_list) == len(legend), "Specify to which legend label each timeseries belongs"
        show_legend = True
    else:
        legend = ['' for ts in range(len(ts_list))]
        show_legend = False


    ts_plot = make_subplots(specs=[[{"secondary_y": True}]]) # create chart
    for ts, ax, leg in zip(ts_list, primary_axis, legend):
        #print(isinstance(ts.index, pd.DatetimeIndex))
        assert isinstance(ts.index, pd.DatetimeIndex), "px series must have a datetime index"
        sampled_ts = ts.iloc[::sample_size]
        ts_plot.add_trace(go.Scatter(y=sampled_ts.values, x=sampled_ts.index, name=leg), secondary_y=not ax) # toggle bool with *-1

    ts_plot.update_yaxes(fixedrange= True, secondary_y=True)

    ts_plot.update_layout(title='<b>Sampled mid</b>', showlegend=show_legend, width=width, height=height)
    ts_plot.show()


# Labels
def plot_labels_line(px_ts, labels, title='Labels', width=900, height=500, **kwargs):
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

    fig.update_layout(title=f'<b>{title}</b>', width=width, height=height)
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
    ''' Plot visual insight for labels on x variable (default "trade_len"):
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