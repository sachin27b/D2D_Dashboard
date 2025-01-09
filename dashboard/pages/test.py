# import dash
# from dash import Input, Output, State, dcc, html
# import dash_bootstrap_components as dbc
# import time
# from threading import Thread
# from dash.long_callback import DiskcacheLongCallbackManager
# import diskcache
# dash.register_page(__name__, path='/test')


# layout = html.Div([
#     dcc.Store(id="progress-store", data=0),
#     dbc.Button("Start Process", id="start-button", n_clicks=0, color="primary"),
#     html.Br(),
#     html.Br(),
#     dbc.Progress(id="progress-bar", striped=True, animated=True, value=0, label="0%"),
#     html.Div(id="output-container"),
# ])

# @dash.long_callback(
#     Output("progress-store", "data"),
#     Input("start-button", "n_clicks"),
#     prevent_initial_call=True,
#     running=[
#         (Output("start-button", "disabled"), True, False),
#     ],
# )
# def start_long_process(n_clicks, progress=Output("progress-store", "data")):
#     for i in range(101):
#         time.sleep(0.1)
#         progress.send(i)
#     return 100

# @dash.callback(
#     [Output("progress-bar", "value"), Output("progress-bar", "label")],
#     Input("progress-store", "data"),
# )
# def update_progress_bar(progress):
#     return progress, f"{progress}%"
