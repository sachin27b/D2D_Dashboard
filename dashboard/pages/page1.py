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

box_style = {'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
            'margin': '10px',
            'flex': '1',
            'textAlign': 'left'
            }


# Define layout
layout = html.Div([
    html.Iframe(
        src="/assets/index.html",  # Path to the HTML file
        style={"width": "100%", "height": "600px", "border": "none"}
    )
])

