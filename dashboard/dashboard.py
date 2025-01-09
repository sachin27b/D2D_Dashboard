import dash
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from components import sidebar_comp
from components import helper,get_artifact
from dotenv import load_dotenv
import os
import time
from dash import Dash, DiskcacheManager, Input, Output, html, callback
import diskcache
from dash.long_callback import DiskcacheLongCallbackManager


def run_dashboard(helper_obj,artifact_obj, app_title='Data Dashboard', debug=True):
    

    COLOR_PALETTE = {
        'primary_dark': '#0A1128',
        'primary_mid': '#1B2845',
        'accent_blue': '#1CB5E0',
        'accent_green': '#5CDB95',
        'text_light': '#F7F9FB',
        'text_subtle': '#B0BEC5'
    }


    content_style = {
        'margin-left': '260px',
        'margin-right': '20px',
        'padding': '30px',
        'backgroundColor': COLOR_PALETTE['text_light']
    }
    # cache = diskcache.Cache("./cache")
    # # background_callback_manager = DiskcacheManager(cache)
    # long_callback_manager = DiskcacheLongCallbackManager(cache)

                
# Initialize the Dash app
# app = Dash(__name__, )
    # Initialize the Dash app
    app = dash.Dash(__name__, 
        external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap'
        ],
        use_pages=True,
        pages_folder="pages"
    )

    
    sidebar = sidebar_comp.create_collapsible_sidebar(helper_obj, 'Menu', COLOR_PALETTE)


    app.layout = html.Div([
        sidebar,
        html.Div([
            html.H3(
                "Dashboard",
                style={
                    'color': COLOR_PALETTE['primary_mid'],
                    'textAlign': 'center',
                    'marginBottom': '30px',
                    'fontWeight': '900',
                    'fontSize': '5rem',
                    'textShadow': '0 0 10px rgba(255, 255, 255, 0.7), 0 0 20px ' + COLOR_PALETTE['primary_mid'],
                    'transition': 'text-shadow 0.3s ease-in-out'
                }
            ),
            html.Img(
            src='assets/logo.png',
            style={
                'height': '80px',
                'position': 'absolute',
                'top': '15px',
                'right': '45px'
            }
        ),
        dash.page_container],
          style={
            **content_style,
            'background': f'linear-gradient(135deg, {COLOR_PALETTE["text_light"]} 0%, #F0F4F8 100%)',
            'borderRadius': '15px',
            'boxShadow': '0 10px 25px rgba(0,0,0,0.1)',
            'padding': '20px'
        })
    ], style={
        'background': f'linear-gradient(135deg, {COLOR_PALETTE["primary_dark"]} 0%, {COLOR_PALETTE["primary_mid"]} 100%)',
        'minHeight': '100vh',
        'margin': '10px',
        'padding': '0',
        'fontFamily': 'Inter, sans-serif'
    })
    
    # @app.callback(
    #     Output('sidebar', 'style'),
    #     Input('sidebar-toggle-btn', 'n_clicks'),
    #     State('sidebar', 'style')
    # )

    # def toggle_sidebar(n_clicks, current_style):
    #     if current_style is None:
    #         current_style = {'transform': 'translateX(0)'}  

    #     if n_clicks is None:
    #         return current_style

    #     if current_style.get('transform', 'translateX(0)') == 'translateX(0)':
    #         return {**current_style, 'transform': 'translateX(-70%)'}
    #     else:
    #         return {**current_style, 'transform': 'translateX(0)'}

    app.run(debug=debug)

if __name__ == '__main__':

    load_dotenv(dotenv_path='../config/.env')
    os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
    os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

    helper_obj = helper.Helper()
    artifact_obj = get_artifact.DownloadArtifact(helper_obj)
    
    run_dashboard(helper_obj,artifact_obj, app_title='Dashboard',debug=True)