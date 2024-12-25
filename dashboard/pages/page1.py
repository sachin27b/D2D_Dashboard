import dash
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from components import helper, get_artifact
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='../config/.env')
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")
helper_obj = helper.Helper()
artifact_obj = get_artifact.DownloadArtifact(helper_obj)

COLOR_PALETTE = {
        'primary_dark': '#0A1128',
        'primary_mid': '#1B2845',
        'accent_blue': '#1CB5E0',
        'accent_green': '#5CDB95',
        'text_light': '#F7F9FB',
        'text_subtle': '#B0BEC5'
    }

dash.register_page(__name__, path='/')


# features = helper_obj.run.data.params['features'].replace('"','').replace("'",'').replace('[','').replace(']','').replace(",",'').split()
# num_columns = len(features)
# # column_names = features
# # print(features)
# num_rows = helper_obj.run.data.params['row_count']


# df = artifact_obj.get_first_five_rows()
# first_five_rows_table = html.Table([
#     html.Thead(
#         html.Tr([html.Th(col, style={'padding': '10px', 'backgroundColor': COLOR_PALETTE['accent_blue'], 'color': 'white'}) for col in df.columns])
#     ),
#     html.Tbody([
#         html.Tr([
#             html.Td(df.iloc[row][col], style={'padding': '8px', 'border': '1px solid #ddd','color': 'darkblue'}) 
#             for col in df.columns
#         ]) for row in range(5)
#     ])
# ], style={
#     'width': '100%', 
#     'borderCollapse': 'collapse', 
#     'marginBottom': '20px',
#     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
# })

# missing_values = artifact_obj.get_null_count()
# missing_values_table = html.Table([
#     html.Thead(
#         html.Tr([
#             html.Th(col, style={'padding': '10px', 'backgroundColor': 'teal', 'color': 'white'}) 
#             for col in missing_values.columns
#         ])
#     ),
#     html.Tbody([
#         html.Tr([
#             html.Td(missing_values.iloc[row][col], style={'padding': '8px', 'border': '1px solid #ddd', 'color': 'darkblue'}) 
#             for col in missing_values.columns
#         ]) for row in range(min(5, len(missing_values)))  
#     ])
# ], style={
#     'width': '100%',
#     'borderCollapse': 'collapse',
#     'marginBottom': '20px',
#     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
# })

# column_names = html.Table([
#         # Table header
#         html.Thead(html.Tr([html.Th("Name",style={'padding': '10px', 'backgroundColor': 'teal', 'color': 'white'})])),
#         # Table body
#         html.Tbody([
#             html.Tr([html.Td(item,style={'padding': '8px', 'border': '1px solid #ddd', 'color': 'darkblue'})]) for index, item in enumerate(features)
#         ])
#     ], style={
#     'width': '100%',
#     'borderCollapse': 'collapse',
#     'marginBottom': '20px',
#     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
# })

# # Combine dataset details into a list of info boxes
# dataset_info = {
#     'Total Columns': num_columns,
#     'Total Rows': num_rows,
#     # 'Column Names': ', '.join(column_names)
# }

# # Null value counts as another box

# # Create info boxes for dataset details
# info_boxes = [
#     html.Div(
#         [
#             html.H3(stat_name, style={
#                 'color': COLOR_PALETTE['primary_mid'], 
#                 'fontSize': '16px', 
#                 'marginBottom': '10px',
#                 'fontWeight': 500
#             }),
#             html.P(str(stat_value), style={
#                 'color': COLOR_PALETTE['accent_blue'], 
#                 'fontSize': '18px', 
#                 'fontWeight': 700,
#                 'margin': '0',
#                 'overflowWrap': 'break-word'
#             })
#         ],
#         style={
#             'backgroundColor': 'white',
#             'padding': '20px',
#             'borderRadius': '12px',
#             'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
#             'margin': '10px',
#             'flex': '1',
#             'textAlign': 'center',
#             'transition': 'transform 0.3s ease',
#             'hover': {'transform': 'translateY(-50px)'}
#         }
#     ) for stat_name, stat_value in dataset_info.items()
# ]

# layout = html.Div([
#         html.H1(
#             f'Data Overview',
#             style={
#                 'color': COLOR_PALETTE['primary_mid'],
#                 'textAlign': 'left',
#                 'marginBottom': '30px',
#                 'fontWeight': '600',
#                 'fontSize': '3rem',
#                 'textShadow': '0 0 10px rgba(255, 255, 255, 0.7), 0 0 20px ' + COLOR_PALETTE['primary_mid'],
#                 'transition': 'text-shadow 0.3s ease-in-out'
#             }
#         ),
#         # Dataset info boxes
#         html.Div(
#             info_boxes,
#             style={
#                 'display': 'flex',
#                 'flexWrap': 'wrap',
#                 'justifyContent': 'space-between',
#             }
#         ),
#         html.Div([
#             html.H3(
#                 'Column Names', 
#                 style={
#                     'color': COLOR_PALETTE['primary_mid'],
#                     'textAlign': 'center',
#                     'marginBottom': '15px'
#                 }
#             ),
#             column_names
#         ], style={
#             'maxHeight': '150px',
#             'overflowY': 'auto',
#             'border': '1px solid #ddd',
#             'marginTop': '30px',
#             'backgroundColor': 'white',
#             'padding': '20px',
#             'borderRadius': '12px',
#             'boxShadow': '0 6px 12px rgba(0,0,0,0.1)'
#         }),
#         # First 5 rows table
#         html.Div([
#             html.H3(
#                 'First 5 Rows', 
#                 style={
#                     'color': COLOR_PALETTE['primary_mid'],
#                     'textAlign': 'center',
#                     'marginBottom': '15px'
#                 }
#             ),
#             first_five_rows_table
#         ], style={
#             'maxHeight': '150px',  # Adjust height as needed
#             'overflowY': 'auto',
#             'border': '1px solid #ddd',
#             'marginTop': '30px',
#             'backgroundColor': 'white',
#             'padding': '20px',
#             'borderRadius': '12px',
#             'boxShadow': '0 6px 12px rgba(0,0,0,0.1)'
#         }),

#         html.Div([
#             html.H3(
#                 'Missing Value Count', 
#                 style={
#                     'color': COLOR_PALETTE['primary_mid'],
#                     'textAlign': 'center',
#                     'marginBottom': '15px'
#                 }
#             ),
#             missing_values_table
#         ], style={
#             'maxHeight': '150px',
#             'overflowY': 'auto',
#             'border': '1px solid #ddd',
#             'marginTop': '30px',
#             'backgroundColor': 'white',
#             'padding': '20px',
#             'borderRadius': '12px',
#             'boxShadow': '0 6px 12px rgba(0,0,0,0.1)'
#         })
#     ],
#     style={
#         'backgroundColor': 'white',
#         'padding': '40px',
#         'borderRadius': '15px',
#         'boxShadow': '0 15px 30px rgba(0,0,0,0.1)',
#         'width': '90%',
#         'margin': 'auto'
#     }
# )

box_style = {'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
            'margin': '10px',
            'flex': '1',
            'textAlign': 'left'
            }


# Define layout
layout = html.Div(
    [
        # Header
        html.H1("D2D Overview", style={
            "textAlign": "left",
            "color": "#0A1128",
            "marginBottom": "20px"
        }),

        # Section 1: D2D - What?
        html.Div([
            # html.H2("What?", style={"color": "#1B2845", "marginBottom": "10px"}),
            html.P(
                "D2D is a statistical measure that calculates the likelihood of default for a loan "
                "by assessing its 'distance' from the upper 4-sigma limit. It generally lies within 0 to 8."
                "The closer the D2D value is to 0, the higher the potential for the account to become a "
                "Non-Performing Asset (NPA).",
                style={'textAlign': 'center',"lineHeight": "1.6", "color": "#333"}
            )
        ], style=box_style),

        # Section 2: D2D - How?
        html.Div([
            html.H2("Methodology", style={"textAlign":"left","color": "#1B2845", "marginBottom": "10px"}),
            html.Ol([
                html.Li("PD Calculation"),
                html.Li("D2D Calculation"),
                html.Li("Recovery Rate Prediction"),
                html.Li("LGD Calculation"),
                html.Li("Capital Charge Calculation using PD and LGD")
            ], style={"lineHeight": "1.8", "color": "#333"})
        ], style=box_style),

        # Section 3: PD - What?
        html.Div([
            html.H2("PD (Probability to Default)", style={"textAlign":"left","color": "#1B2845", "marginBottom": "10px"}),
            html.P(
                "The Probability of Default (PD) is defined as 'the degree of likelihood that the borrower of "
                "a loan or debt will not be able to make the necessary scheduled repayments.' The higher the "
                "probability, the greater the likelihood that a customer defaults in paying obligations to the bank.",
                style={"lineHeight": "1.6", "color": "#333"}
            ),
            html.P(
                "Uses historical loan data and key factors to predict PD using statistical models.",
                style={"lineHeight": "1.6", "color": "#333"}
            )
        ], style=box_style),

        # Section 4: LGD - What?
        html.Div([
            html.H2("LGD (Loss Given Default)", style={"color": "#1B2845", "marginBottom": "10px"}),
            html.P(
                "Loss Given Default (LGD) can be defined as the amount of loss a bank or a financial institution "
                "may suffer on default of a particular facility. LGD is also expressed and interpreted as a "
                "percentage of exposure at default (EAD).",
                style={"lineHeight": "1.6", "color": "#333"}
            ),
            # html.H4("How?", style={"marginTop": "10px", "color": "#1B2845"}),
            html.P(
                "Uses statistical models and historical data to estimate recovery rates.",
                style={"lineHeight": "1.6", "color": "#333"}
            )
        ], style=box_style),


    ],
    style={
        "padding": "30px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#F7F9FB",
        "color": "#333"
    }
)

