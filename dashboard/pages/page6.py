import dash
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from components import helper, get_artifact
import shutil

from dotenv import load_dotenv
import os
import json
import dash_daq as daq


load_dotenv(dotenv_path='../config/.env')
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

# Enhanced Color Palette with Modern Design
COLOR_PALETTE = {
    'primary_dark': '#0A1128',
    'primary_mid': '#1B2845',
    'accent_blue': '#1CB5E0',
    'accent_green': '#5CDB95',
    'text_light': '#F7F9FB',
    'text_subtle': '#B0BEC5'
}

dash.register_page(__name__, path='/d2d')

airb = pd.read_csv("assets/airb.csv")
firb = pd.read_csv("assets/firb.csv")

airb_table = html.Table([
    html.Thead(
        html.Tr([html.Th(col, style={'padding': '10px', 'backgroundColor': COLOR_PALETTE['accent_blue'], 'color': 'white'}) for col in airb.columns])
    ),
    html.Tbody([
        html.Tr([
            html.Td(airb.iloc[row][col], style={'padding': '8px', 'border': '1px solid #ddd','color': 'darkblue'}) 
            for col in airb.columns
        ]) for row in range(500)
    ])
], style={
    'width': '100%', 
    'borderCollapse': 'collapse', 
    'marginBottom': '20px',
    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
})

firb_table = html.Table([
    html.Thead(
        html.Tr([html.Th(col, style={'padding': '10px', 'backgroundColor': COLOR_PALETTE['accent_blue'], 'color': 'white'}) for col in firb.columns])
    ),
    html.Tbody([
        html.Tr([
            html.Td(firb.iloc[row][col], style={'padding': '8px', 'border': '1px solid #ddd','color': 'darkblue'}) 
            for col in firb.columns
        ]) for row in range(len(firb))
    ])
], style={
    'width': '100%', 
    'borderCollapse': 'collapse', 
    'marginBottom': '20px',
    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
})
layout = html.Div([
        
        html.Div([
            html.H3(
                'A-IRB', 
                style={
                    'color': COLOR_PALETTE['primary_mid'],
                    'textAlign': 'center',
                    'marginBottom': '15px'
                }
            ),
            airb_table
        ], style={
            'maxHeight': '250px',  # Adjust height as needed
            'overflowY': 'auto',
            'border': '1px solid #ddd',
            'marginTop': '30px',
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 6px 12px rgba(0,0,0,0.1)'
        }),
        html.Div([
            html.H3(
                'F-IRB', 
                style={
                    'color': COLOR_PALETTE['primary_mid'],
                    'textAlign': 'center',
                    'marginBottom': '15px'
                }
            ),
            firb_table
        ], style={
            'maxHeight': '250px',  # Adjust height as needed
            'overflowY': 'auto',
            'border': '1px solid #ddd',
            'marginTop': '30px',
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 6px 12px rgba(0,0,0,0.1)'
        })
        ],

    style={
        'backgroundColor': 'white',
        'padding': '40px',
        'borderRadius': '15px',
        'boxShadow': '0 15px 30px rgba(0,0,0,0.1)',
        'width': '90%',
        'margin': 'auto'
    }
)