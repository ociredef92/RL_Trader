"""
Future improvs
Go back to original files, export on the fly with preprocessing
Chart at different time frequencies

Write support functions and class (to open files as well - or can just import the file)
merge first 2 plot in 1 initial plot
fix slider
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash.dependencies import Input, Output

from layout import user_interface

import func_tools as ft

root_caching_folder = "" # processed cached data folder

# Define app and app layout
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = user_interface


@app.callback(
    [Output("price_chart", "figure"),
    Output("print_mom_window", "children"),
    Output("print_alpha_threshold", "children"),
    Output("print_tr_cost_bps", "children")
    ]
    ,
    [
        Input("ccy_code", "value"),
        #Input("chart_type_radio", "value"),
        Input("price_date_range", "start_date"),
        Input("price_date_range", "end_date"),

        Input("normalization_type", "value"),
        Input("normalization_window", "value"),
        Input("label_switch_on", "value"),
        Input("momentum_window", "value"),
        Input("alpha_threshold", "value"),
        Input("tr_costs", "value"),
        Input("ls_pnl", "value")
    ],
)
def make_price_graph(security, start_date, end_date, norm_type, norm_window, switch, k, alpha, tr_costs, pnl):

    # initiate values to print out under dash components
    mom_window_text = ''
    alpha_thresh_text = ''
    tr_costs_text = ''
    
    # data reading
    bbo_df = pd.read_csv(f'{root_caching_folder}{security}/bbo.csv', header=0, index_col=0)
    #bbo_df.index = bbo_df.index.set_names(['date'])
    #bbo_df = bbo_df.reset_index()
    print(bbo_df.tail(10))
    bbo_filtered = bbo_df[(bbo_df.index >= start_date) & (bbo_df.index <= end_date) ]
    print(security)

    #if chart_type == 'Candlestick':

    #    px_chart = go.Figure(data=[go.Candlestick(x=bbo_filtered['date'],
    #                open=bbo_filtered['mid_open'], high=bbo_filtered['mid_high'],
    #                low=bbo_filtered['mid_low'], close=bbo_filtered['mid_close'])
    #                ])
        
    #elif chart_type == 'Line':

    #    px_chart.add_trace(go.Scatter(x=bbo_filtered['date'], y=bbo_filtered['mid_mean']), secondary_y=False)

    #print(norm_window)
    
    px_chart = make_subplots(rows=2, cols=1, subplot_titles=("", "Theoretical PnL"), shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing = 0.20, 
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

    #for i in range(len(norm_type)):
    sec_axis_check = False
    if 'price' in norm_type:
        px_chart.add_trace(go.Scatter(x=bbo_filtered.index, y=bbo_filtered['mid_mean'], name='price'), 
                            row=1, col=1, secondary_y=False)
        px_chart.update_yaxes(title_text="$ price", secondary_y=sec_axis_check, row=1, col=1)
        sec_axis_check = True

    if 'z_score' in norm_type:
        norm_ts = ft.normalize(bbo_filtered['mid_mean'], norm_type='z_score')
        px_chart.add_trace(go.Scatter(x=bbo_filtered.index, y=norm_ts, name='z-score'), 
                            row=1, col=1, secondary_y=sec_axis_check)
        px_chart.update_yaxes(title_text="normalized price", secondary_y=sec_axis_check, row=1, col=1)

    if 'dyn_z_score' in norm_type:
        norm_ts = ft.normalize(bbo_filtered['mid_mean'], norm_type='dyn_z_score', roll=norm_window)
        px_chart.add_trace(go.Scatter(x=bbo_filtered.index, y=norm_ts, name='dynamic z-score'), 
                            row=1, col=1, secondary_y=sec_axis_check)
        px_chart.update_yaxes(title_text="normalized price", secondary_y=sec_axis_check, row=1, col=1)

    if switch == 'on':
        
        labels = ft.get_labels(bbo_filtered['mid_mean'], k, alpha) #getting labels from real px
        background_color = ft.plot_labels(labels)
        px_chart.update_layout(shapes=background_color) # plot labels background

        # If labelling is on plot theoretical pnl
        if pnl == 'long':
            pnl = ft.get_pnl(bbo_filtered['mid_mean'], labels, long_only=True)
            px_chart.add_trace(go.Scatter(x=pnl.index, y=pnl, name='PnL'), row=2, col=1)

        elif pnl == 'long_short':
            pnl = ft.get_pnl(bbo_filtered['mid_mean'], labels, long_only=False)
            px_chart.add_trace(go.Scatter(x=pnl.index, y=pnl, name='PnL'), row=2, col=1)

    else:
        # add flat line
        px_chart.add_trace(go.Scatter(x=bbo_filtered.index, y=np.zeros(bbo_filtered.index.shape[0]), name='PnL'), row=2, col=1)

    # print out labelling paramenters
    mom_window_text = f'Momentum Window: {k} steps'
    alpha_thresh_text = f'Alpha Threshold: {alpha*100:.3f}%'

    # print out transaction cost assumption
    tr_costs_text = f'Transaction Costs: {tr_costs}bps'

    px_chart.update_layout(
                height=600,
                title=f'<b>{security} pair</b>',
                title_x=0.5,
                #yaxis_title="$ price",
                legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.0,
                            xanchor="left",
                            x=0
                ),
                xaxis_showticklabels=True, 
                xaxis2_showticklabels=True
            )
    #px_chart.update_xaxes(rangeslider_visible=True)

    return px_chart, mom_window_text, alpha_thresh_text, tr_costs_text

# Deactivate normalization slider when dyn_z_score is not selected
@app.callback(Output('normalization_window', 'disabled'),
             [Input('normalization_type', 'value')])
def disable_norm(value_list):
    #print(value_list)
    if  "dyn_z_score" not in value_list:
        return True
    else:
        return False

# Deactivate labelling parameters if switch is "off"
@app.callback([
            Output('momentum_window', 'disabled'),
            Output('alpha_threshold', 'disabled')],
            [Input('label_switch_on', 'value')] )
def disable_label(switch):
    if switch == 'off':
        return True, True
    else:
        return False, False


@app.callback(
    Output("depth_chart", "figure"),
    [
        Input("ccy_code", "value"),
        Input("price_date_range", "start_date"),
        Input("price_date_range", "end_date")
    ],
)
def make_depth(security, start_date, end_date):

    depth_df = pd.read_csv(f'{root_caching_folder}{security}/depth.csv', header=0, index_col=0)
    depth_df = depth_df[(depth_df.index >= start_date) & (depth_df.index <= end_date) ]

    d_chart = go.Figure()
    
    x = depth_df.index

    d_chart.add_trace(go.Scatter(x=x, y=depth_df['bid_tight_depth'], fill='tozeroy', name='Bid depth <= 25bps',
                            legendgroup='bid', marker=dict(color='#81C342'))) # fill down to xaxis

    d_chart.add_trace(go.Scatter(x=x, y=depth_df['bid_medium_depth'], fill='tonexty', name='Bid depth <= 50bps',
                            legendgroup='bid',marker=dict(color='#73A541'))) # fill to trace0 y

    d_chart.add_trace(go.Scatter(x=x, y=depth_df['bid_wide_depth'], fill='tonexty', name='Bid depth <= 100bps',
                            legendgroup='bid', marker=dict(color='#69913D'))) # fill to trace0 y

    d_chart.add_trace(go.Scatter(x=x, y=-depth_df['ask_tight_depth'], fill='tozeroy', name='Ask depth <= 25bps',
                            legendgroup='ask',marker=dict(color='#EB2030')))# fill down to xaxis

    d_chart.add_trace(go.Scatter(x=x, y=-depth_df['ask_medium_depth'], fill='tonexty',name='Ask depth <= 50bps',
                            legendgroup='ask', marker=dict(color='#C6272E'))) # fill to trace0 y

    d_chart.add_trace(go.Scatter(x=x, y=-depth_df['ask_wide_depth'], fill='tonexty', name='Ask depth <= 100bps',
                            legendgroup='ask', marker=dict(color='#972B2A'))) # fill to trace0 y

    d_chart.update_layout(legend={'traceorder':'grouped', 'y':0.5}, title='<b>Order Book Depth</b>', title_x=0.5)

    return d_chart

if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1', port='8050')