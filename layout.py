import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime, timedelta

security_list = ['USDT_BTC', 'MOCK_SIN_1']

controls_px = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Select Currency Pair"),
                dcc.Dropdown(
                    id="ccy_code",
                    options=[
                        {"label": col, "value": col} for col in security_list
                    ],
                    value='USDT_BTC',
                ),
            ]
        ),

        #dbc.FormGroup(
        #    [
        #        dbc.Label("Select Chart Type"),
        #        dcc.RadioItems(
        #            id='chart_type_radio',
        #            options=[
        #                {'label': 'Candlestick', 'value': 'Candlestick'},
        #                {'label': 'Line', 'value': 'Line'},
        #            ],
        #            value='Line',
        #            labelStyle={'display': 'inline-block'},
        #        )

        #    ]
        #),

        dbc.FormGroup(
            [
                dbc.Label("Select Date Range"),
                dcc.DatePickerRange(
                    id='price_date_range',
                    start_date=datetime(2020,5,1),
                    end_date=datetime(2020,6,1),
                    display_format='MMM Do, YY'
                )  
            ]
        ),
    ],
    body=True,
)


controls = dbc.Card(
    
    [
        html.H5("Selection"),
        dbc.FormGroup(
            [
                dbc.Label("Currency Pair"),
                dcc.Dropdown(
                    id="ccy_code",
                    options=[
                        {"label": col, "value": col} for col in security_list
                    ],
                    value='USDT_BTC',
                ),
            ]
        ),

        dbc.FormGroup(
            [
                dbc.Label("Date"),
                dcc.DatePickerRange(
                    id='price_date_range',
                    start_date=datetime(2020,5,1),
                    end_date=datetime(2020,6,1),
                    display_format='MMM Do, YY'
                ), 
            ]
        ),

        html.Hr(),
        html.H5("Normalization"),
        dbc.FormGroup(
            [
                dbc.Label(hidden=True),

                dcc.Checklist(
                    id='normalization_type',
                    options=[
                        {'label': 'price', 'value': 'price'},
                        {'label': 'z-score', 'value': 'z_score'},
                        {'label': 'dynamic z-score', 'value': 'dyn_z_score'}
                    ],
                    value=['price'],
                    labelStyle={'display': 'inline-block'}
                ),

                dcc.Slider(
                    id='normalization_window',
                    min=0,
                    max=2880,
                    step=1,
                    value=720,
                    marks={
                            0: {'label': '0'},
                            144: {'label': '1d'},
                            288: {'label': '2d'},
                            720: {'label': '5d'},
                            1440: {'label': '10d'},
                            2880: {'label': '20d'}
                        }
                ),
            ]
        ),

        html.Hr(),
        html.H5("Labelling"),
        dbc.FormGroup(
            [

                dcc.RadioItems(
                    id='label_switch_on',
                    options=[
                        {'label': 'On', 'value': 'on'},
                        {'label': 'Off', 'value': 'off'}
                    ],
                    value='off',
                    labelStyle={'display': 'inline-block'}
                ),

                dbc.Label("Momentum Window", id='print_mom_window'),
                dcc.Slider(
                    id='momentum_window',
                    min=0,
                    max=120,
                    step=1,
                    value=18,
                    marks={
                            0: {'label': '0'},
                            6: {'label': '1h'},
                            12: {'label': '2h'},
                            18: {'label': '3h'},
                            30: {'label': '5h'},
                            60: {'label': '10h'},
                            120:{'label': '20h'}
                        }
                ),
                #html.P(id='print_mom_window')
            ]
        ),

        dbc.FormGroup(
            [
                dbc.Label("Alpha Threshold", id='print_alpha_threshold'),
                dcc.Slider(
                    id='alpha_threshold',
                    min=0,
                    max=0.02,
                    step=0.00001,
                    value=0.01,
                    marks={
                            0: {'label': '0%'},
                            0.0025: {'label': '0.25%'},
                            0.0050: {'label': '0.5%'},
                            0.010: {'label': '1%'},
                            0.015: {'label': '1.5%'},
                            0.020: {'label': '2%'},
                            #0.030: {'label': '3.00%'},
                            #0.05: {'label': '0.05'},
                        }
                ),
            ]
        ),

        html.Hr(),
        html.H5("Labels Theoretical P&L"),
        dbc.FormGroup(
            [
                dbc.Label(hidden=True),

                dcc.RadioItems(
                    id='ls_pnl',
                    options=[
                        {'label': 'long only', 'value': 'long'},
                        {'label': 'long-short', 'value': 'long_short'}
                    ],
                    value='long',
                    labelStyle={'display': 'inline-block'}
                ),
                dbc.Label("Transaction Costs: ", id='print_tr_cost_bps'),
                dcc.Slider(
                    id='tr_costs',
                    min=0,
                    max=100,
                    step=1,
                    value=20,
                    marks={
                            0: {'label': '0'},
                            20: {'label': '20'},
                            40: {'label': '40'},
                            60: {'label': '60'},
                            80: {'label': '80'},
                            100: {'label': '100'}
                        }
                ),
            ]
        ),


    ],
    body=True,
)

user_interface = dbc.Container(
    [
        html.H1("Crypto Dashboard"),
        html.Hr(),

        dbc.Row(
            [
                dbc.Col([controls], md=4),
                dbc.Col(dcc.Graph(id="price_chart"), md=8),
            ],
                align="start",
        ),

        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="depth_chart"), md=12),
            ],
            align="center",
        ),
                

    ],
    fluid=True,
)