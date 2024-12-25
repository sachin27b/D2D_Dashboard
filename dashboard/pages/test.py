import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import Dash, html
import dash_daq as daq

dash.register_page(__name__, path='/test')



theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

layout = html.Div(
    daq.DarkThemeProvider(
        theme=theme,
        children=html.Div([
            daq.Thermometer(
                min=0,
                max=100,
                value=95.6,
                id='darktheme-daq-thermometer',
                className='dark-theme-control'
            ),
            html.Div(
                    "Temperature: 98.6°F",  # Display the value
                    id='thermometer-value',
                    style={
                        'color': theme['primary'],
                        'textAlign': 'center',
                        'fontSize': '20px',
                        'marginTop': '10px'
                    }
                )   
        ], style={
            'border': 'solid 1px #A2B1C6',
            'border-radius': '5px',
            'padding': '20px',
            'marginTop': '20px',
            # 'backgroundColor': '#303030',  # Dark background color
            'color': '#FFFFFF'  # Text color for readability
        })
    )
)
