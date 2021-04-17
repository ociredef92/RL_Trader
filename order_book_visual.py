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

root_caching_folder = "Processed_Data"#"RL_Trader/Processed_Data" # processed cached data folder

# Define app and app layout
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
security = 'USDT_BTC'
app.layout = user_interface

@app.callback(
    [Output("price_chart", "figure"),
    Output("print_k_plus_window", "children"),
    Output("print_k_minus_window", "children"),
    Output("print_alpha_threshold", "children"),
    Output("print_tr_fee_bps", "children"),
    Output("pnl", "figure")
    ]
    ,
    [
        Input("ccy_code", "value"),
        #Input("chart_type_radio", "value"),
        Input("price_date_range", "start_date"),
        Input("price_date_range", "end_date"),
        Input("ob_levels", "value"),
        #Input("normalization_type", "value"),
        Input("normalization_window", "value"),
        Input("label_switch_on", "value"),
        Input("k_plus", "value"),
        Input("k_minus", "value"),
        Input("alpha_threshold", "value"),
        Input("tr_fee_bps", "value"),
        Input("long_only_pnl", "value")
    ],
)
def make_price_graph(security, start_date, end_date, ob_levels, norm_window, switch, 
                        k_plus, k_minus, alpha, tr_fee_bps, long_only):

    # initiate values to print out under dash components
    k_plus_window_text = ''
    k_minus_window_text = ''
    alpha_thresh_text = ''
    tr_fee_text = ''
    
    # data reading
    data = pd.read_csv(f'{root_caching_folder}/{security}/data-cache-1m.csv', index_col=0)
    data = data[(data.Datetime >= start_date) & (data.Datetime <= end_date)]

    data_top = data[data.Level  == 0]#.reset_index() #fix double index issue. Do it the func tool way, cause that's the one that changes
    data_top['Mid_Price'] = (data_top['Ask_Price'] + data_top['Bid_Price']) / 2
    data_top['Spread'] = (data_top['Ask_Price'] - data_top['Bid_Price']) / data_top['Mid_Price']
    data_grouped = data_top.groupby('Datetime').agg(
        {'Ask_Size':'sum',
        'Bid_Size':'sum',
        'Spread': 'min'
        }
    )
    #bbo_df.index = bbo_df.index.set_names(['date'])
    #bbo_df = bbo_df.reset_index()
    #print(bbo_df.tail(10))
    
    #print(security)
    
    px_chart = make_subplots(rows=2, cols=1, subplot_titles=("", "10 levels depth and spread"), shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing = 0.10, 
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    pnl_chart = px.line(height=200)
    #for i in range(len(norm_type)):
    sec_axis_check = False

    # add depth and spread to main chart
    px_chart.add_trace(go.Scatter(x=data_grouped.index.values, y=data_grouped['Bid_Size'].values,  name='Bid depth - 10 levels',
                                marker=dict(color='#81C342')), row=2, col=1, secondary_y=False) # fill down to xaxis

    px_chart.add_trace(go.Scatter(x=data_grouped.index.values, y=-data_grouped['Ask_Size'].values,  name='Ask depth - 10 levels',
                                marker=dict(color='#EB2030')), row=2, col=1, secondary_y=False) # fill down to xaxis

    px_chart.add_trace(go.Scatter(x=data_grouped.index.values, y=data_grouped['Spread'].values,  name='Best bid-offer spread',
                                marker=dict(color='#335eff')), row=2, col=1, secondary_y=True) # fill down to xaxis



    px_chart.add_trace(go.Scatter(x=data_top['Datetime'], y=data_top['Mid_Price'], name='price', marker=dict(color='#000000')), 
                        row=1, col=1, secondary_y=False)
    px_chart.update_yaxes(title_text="$ price", secondary_y=sec_axis_check, row=1, col=1)
    sec_axis_check = True



    # if 'z_score' in norm_type:
    #     norm_ts = ft.normalize(data[['Ask_Price', 'Bid_Price']], ob_levels=ob_levels, norm_type='z_score')
    #     px_chart.add_trace(go.Scatter(x=data.index, y=norm_ts, name='z-score', marker=dict(color='#19D3F3')), 
    #                         row=1, col=1, secondary_y=sec_axis_check)
    #     px_chart.update_yaxes(title_text="normalized price", secondary_y=sec_axis_check, row=1, col=1)

    #if 'dyn_z_score' in norm_type:
    data_ft = data.set_index(['Datetime', 'Level'])
    norm_ts_px = ft.normalize(data_ft[['Ask_Price', 'Bid_Price']], ob_levels=ob_levels, norm_type='dyn_z_score', roll=norm_window)
    norm_ts_vol = ft.normalize(data_ft[['Ask_Size', 'Bid_Size']], ob_levels=ob_levels, norm_type='dyn_z_score', roll=norm_window) # get norm volumes
    test_dyn_df = pd.concat([norm_ts_px, norm_ts_vol], axis=1).reset_index() # concat along row index
    depth_dyn, dt_index_dyn = ft.reshape_lob_levels(test_dyn_df, output_type='array') # 1 train dataset
    mid_px_train_dyn = pd.Series((depth_dyn[:,2] + depth_dyn[:,0]) / 2) # 2

    px_chart.add_trace(go.Scatter(x=dt_index_dyn, y=mid_px_train_dyn, name='dynamic z-score',  marker=dict(color='#FFA15A')), 
                        row=1, col=1, secondary_y=sec_axis_check)
    px_chart.update_yaxes(title_text="normalized price", secondary_y=sec_axis_check, row=1, col=1)

    if switch == 'on':

        # If labelling is on plot theoretical pnl
        if long_only == 'long':

            labels = ft.get_labels(data['Mid_Price'], k_plus, k_minus,  alpha, long_only=True) #getting labels from real px
            print(labels.shape)
            
            pnl, _ = ft.get_pnl(data['Mid_Price'], labels, tr_fee_bps/10000)
            pnl_chart.add_trace(go.Scatter(x=pnl.index, y=pnl, name='PnL'))

        elif long_only == 'long_short':
            labels = ft.get_labels(data['Mid_Price'], k_plus, k_minus,  alpha, long_only=False) #getting labels from real px
            
            pnl, _ = ft.get_pnl(data['Mid_Price'], labels, tr_fee_bps/10000)
            pnl_chart.add_trace(go.Scatter(x=pnl.index, y=pnl, name='PnL'))

        background_color = ft.plot_labels(labels)
        px_chart.update_layout(shapes=background_color) # plot labels background
    else:
        # add flat line
        pnl_chart.add_trace(go.Scatter(x=data.index, y=np.zeros(data.index.shape[0]), name='PnL'))

    # print out labelling paramenters
    k_plus_text = f'k Plus Window: {k_plus} steps'
    k_minus_text = f'k Minus Window: {k_minus} steps'
    alpha_thresh_text = f'Alpha Threshold: {alpha*100:.3f}%'

    # print out transaction cost assumption
    tr_fee_text = f'Transaction Costs: {tr_fee_bps}bps'

    px_chart.update_layout(
                height=900,
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

    return px_chart, k_plus_text, k_minus_text, alpha_thresh_text, tr_fee_text, pnl_chart

# # Deactivate normalization slider when dyn_z_score is not selected
# @app.callback(Output('normalization_window', 'disabled'),
#              [Input('normalization_type', 'value')])
# def disable_norm(value_list):
#     #print(value_list)
#     if  "dyn_z_score" not in value_list:
#         return True
#     else:
#         return False

# Deactivate labelling parameters if switch is "off"
@app.callback([
            Output('k_plus', 'disabled'),
            Output('k_minus', 'disabled'),
            Output('alpha_threshold', 'disabled')],
            [Input('label_switch_on', 'value')] )
def disable_label(switch):
    if switch == 'off':
        return True, True, True
    else:
        return False, False, False


# @app.callback(
#     Output("depth_chart", "figure"),
#     [
#         Input("ccy_code", "value"),
#         Input("price_date_range", "start_date"),
#         Input("price_date_range", "end_date")
#     ],
# )
# def make_depth(security, start_date, end_date):

#     depth_df = pd.read_csv(f'{root_caching_folder}{security}/depth.csv', header=0, index_col=0)
#     depth_df = depth_df[(depth_df.index >= start_date) & (depth_df.index <= end_date) ]

#     d_chart = go.Figure()
    
#     x = depth_df.index

#     d_chart.add_trace(go.Scatter(x=x, y=depth_df['bid_tight_depth'], fill='tozeroy', name='Bid depth <= 25bps',
#                             legendgroup='bid', marker=dict(color='#81C342'))) # fill down to xaxis

#     d_chart.add_trace(go.Scatter(x=x, y=depth_df['bid_medium_depth'], fill='tonexty', name='Bid depth <= 50bps',
#                             legendgroup='bid',marker=dict(color='#73A541'))) # fill to trace0 y

#     d_chart.add_trace(go.Scatter(x=x, y=depth_df['bid_wide_depth'], fill='tonexty', name='Bid depth <= 100bps',
#                             legendgroup='bid', marker=dict(color='#69913D'))) # fill to trace0 y

#     d_chart.add_trace(go.Scatter(x=x, y=-depth_df['ask_tight_depth'], fill='tozeroy', name='Ask depth <= 25bps',
#                             legendgroup='ask',marker=dict(color='#EB2030')))# fill down to xaxis

#     d_chart.add_trace(go.Scatter(x=x, y=-depth_df['ask_medium_depth'], fill='tonexty',name='Ask depth <= 50bps',
#                             legendgroup='ask', marker=dict(color='#C6272E'))) # fill to trace0 y

#     d_chart.add_trace(go.Scatter(x=x, y=-depth_df['ask_wide_depth'], fill='tonexty', name='Ask depth <= 100bps',
#                             legendgroup='ask', marker=dict(color='#972B2A'))) # fill to trace0 y

#     d_chart.update_layout(legend={'traceorder':'grouped', 'y':0.5}, title='<b>Order Book Depth</b>', title_x=0.5)

#     return d_chart

if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1', port='8050')