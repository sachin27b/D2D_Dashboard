import dash
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State

def create_collapsible_sidebar(df, app_title, COLOR_PALETTE):


    sidebar_style = {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'width': '240px',
        'padding': '30px 15px',
        'background': f'linear-gradient(135deg, {COLOR_PALETTE["primary_dark"]} 0%, {COLOR_PALETTE["primary_mid"]} 100%)',
        'box-shadow': '4px 0 15px rgba(0,0,0,0.2)',
        'z-index': 1000,
        'fontFamily': 'Inter, sans-serif'
    }

    sidebar_button_style = {
        'padding': '12px 15px', 
        'margin': '10px 0', 
        'background': f'linear-gradient(90deg, {COLOR_PALETTE["accent_blue"]} 0%, {COLOR_PALETTE["accent_green"]} 100%)', 
        'border-radius': '8px',
        'text-align': 'center',
        'color': COLOR_PALETTE['text_light'],
        'fontWeight': 600,
        'transition': 'all 0.3s ease',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
        'border': 'none'
    }
    # Sidebar button style
    # sidebar_button_style = {
    #     'padding': '10px 15px',
    #     'margin': '10px 0',
    #     'backgroundColor': COLOR_PALETTE['primary_dark'],
    #     'color': COLOR_PALETTE['text_light'],
    #     'border': 'none',
    #     'borderRadius': '5px',
    #     'cursor': 'pointer',
    #     'transition': 'all 0.3s ease'
    # }

    # # Sidebar style
    # sidebar_style = {
    #     'position': 'fixed',
    #     'top': 0,
    #     'left': 0,
    #     'bottom': 0,
    #     'width': '250px',
    #     'padding': '20px',
    #     'backgroundColor': COLOR_PALETTE['background_dark'],
    #     'transition': 'transform 0.3s ease',
    # }

    sidebar_layout = html.Div([
        # Collapse/Expand Button
        html.Button(
            '✦', 
            id='sidebar-toggle-btn',
            className='sidebar-toggle-btn',
            # style={
            #     'position': 'absolute', 
            #     'top': '10px', 
            #     'right': '10px', 
            #     'background': 'none', 
            #     'border': 'none', 
            #     'fontSize': '25px', 
            #     'color': COLOR_PALETTE['text_light'],
            #     'cursor': 'pointer',
            #     'zIndex': 1000
            # }
        ),
        
        # Sidebar Content
        html.Div([
            html.H2(app_title, style={
                'fontFamily': 'Inter, sans-serif',
                'color': COLOR_PALETTE['text_light'], 
                'textAlign': 'center', 
                'fontWeight': 700,
                'marginBottom': '30px',
                'textShadow': '0 0 10px rgba(255, 255, 255, 0.8)'  # Glowing effect
                
            }),
            html.Hr(style={'borderColor': COLOR_PALETTE['text_subtle'], 'marginBottom': '20px'}),
            
            # Navigation Links
            dcc.Link(
                html.Div('D2D Overview', style=sidebar_button_style), 
                href='/',
                className='sidebar-link'
            ),
            dcc.Link(
                html.Div('Dataset Overview', style=sidebar_button_style), 
                href='/dataset-overview',
                className='sidebar-link'
            ),
            dcc.Link(
                html.Div('Model', style=sidebar_button_style), 
                href='/model',
                className='sidebar-link'
            ),
            dcc.Link(
                html.Div('What If Analysis', style=sidebar_button_style),
                href='/what-if',
                className='sidebar-link'
            ),
            dcc.Link(
                html.Div('Backtesting', style=sidebar_button_style),
                href='/backtesting',
                className='sidebar-link'
            )
        ], id='sidebar-content')
    ], 
    id='sidebar', 
    style=sidebar_style)

    return sidebar_layout