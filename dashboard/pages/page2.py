'''
TODO in this page:
1. numerical column dropdown to show plots
2. categorical column dropdown to show plots
3. multivariate plots dropdown'''
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import os

dash.register_page(__name__, path='/dataset-overview')

COLOR_PALETTE = {
    'primary_dark': '#0A1128',
    'primary_mid': '#1B2845',
    'accent_blue': '#1CB5E0',
    'accent_green': '#5CDB95',
    'text_light': '#F7F9FB',
    'text_subtle': '#B0BEC5'
}

num_directory = "assets/plots/numerical_columns"
num_files = [f for f in os.listdir(num_directory) if f.endswith(".csv")]

cat_directory = "assets/plots/categorical_columns"
cat_files = [f for f in os.listdir(cat_directory) if f.endswith(".csv")]

multi_directory = "assets/plots/multivariate"
multi_files = [f for f in os.listdir(multi_directory) if f.endswith(".csv")]

def plot_corr(path):
    correlation_matrix = pd.read_csv(path)

    heatmap_data = correlation_matrix.pivot(
        index='X', columns='Y', values=correlation_matrix.columns[2])

    corr = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        colorbar=dict(title='Correlation')
    ))

    corr.update_layout(
        # title="Heatmap of Correlation Between Independent Variables",
        xaxis_title="Independent Variables",
        yaxis_title="Independent Variables",
        height=900,
    )
    return corr


annotation = {
    'age': (
        "From the histogram below it can be seen that highest number of borrowers are from age 47 to 59<br>"
    ),
    'sanctioned_loan_amount': (
        "From the below histogram it can be seen that approximately 10 lakh customers have been sanctioned loan below 2.5 lakhs."
    ),
    'balance': ("From the below histogram it can be seen that approximately 5.4 lakh borrowers have less than 1 lakh outstanding balance. <br>"
                "About 4 lakh borrowers have outstanding balance > 1 lakh and < 2 lakh."),
    'GL_product': (""),
    'interest_rates': (""),
    'product': (""),
    'unsecured_loan': (""),
    'used_rv': (""),
    'activity': (""),
    'area': (""),
    'caste': (""),
    'constitution': (""),
    'customer_asset_class': (""),
    'customer_status': (""),
    'district': (""),
    'facility_type': (""),
    'occupation': (""),
    'scheme': (""),
    'subsector': ("")
}

h1_style = {
    'color': COLOR_PALETTE['primary_mid'],
    'textAlign': 'left',
    'marginBottom': '20px',
    'fontWeight': '600',
    'fontSize': '3rem',
                'textShadow': '0 0 10px rgba(255, 255, 255, 0.7), 0 0 20px ' + COLOR_PALETTE['primary_mid']}

h3_style = {
    'color': COLOR_PALETTE['accent_blue'],
    'fontSize': '18px',
    'fontWeight': 700,
    'margin': '0',
    'overflowWrap': 'break-word'
}

p_style = {
    'color': COLOR_PALETTE['primary_mid'],
    'fontSize': '16px',
    'marginBottom': '10px',
    'fontWeight': 500
}
box_style = {
    'backgroundColor': 'white',
    'padding': '20px',
    'borderRadius': '12px',
                    'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
                    'margin': '10px',
                    'flex': '1',
                    'textAlign': 'center'
}


def plot_line_graph(name,y_label,x_label):
    data = pd.read_csv(f'assets/plots/{name}.csv')
    data = data.sort_values(by=data.columns[0])
    line_fig = px.line(
        data,
        x=data.columns[0],  # Use the first column as X
        y=data.columns[1],  # Use the second column as Y
        # title="Line Graph",
        labels={
            data.columns[0]: x_label,  # Label for X-Axis
            data.columns[1]: y_label  # Label for Y-Axis
        },
        # Optional: Set the style (e.g., 'plotly', 'ggplot2')
        template='plotly'
    )

    return line_fig


def plot_bar_graph(name: str):

    df = pd.read_csv(f'assets/plots/{name}.csv')
    fig = px.histogram(df, x=df[df.columns[0]],
                       y=df[df.columns[1]], color_discrete_sequence=['turquoise'])

    # fig.add_trace(marker_color='#87EBA4')
    # Define the layout
    fig.update_layout(
        bargap=0.1,
        title=f"Histogram for {df.columns[0]}",
        xaxis=dict(title=f"{df.columns[0]}"),
        yaxis=dict(title="Count"),
        template='ggplot2',

    )
    fig.update_xaxes(categoryorder='total descending')

    return fig


def plot_pie_chart(name: str):
    # Replace with your actual CSV file path
    data = pd.read_csv(f'assets/plots/{name}.csv')

    # Create a pie chart
    pie_fig = px.pie(
        data,
        names=data.columns[0],  # Use the first column as the category labels
        values=data.columns[1],  # Use the second column as the values (counts)
        # title="Pie Chart",
        template='plotly',
        color_discrete_sequence=px.colors.sequential.Teal
    )
    return pie_fig


layout = html.Div(
    [
        html.Div([
            html.H1("Dataset Overview", style=h1_style)
        ]),
        html.Div([
            html.Div(
                [
                    html.H3("Default Rate", style=h3_style),
                    html.P("14.02%", style=p_style)
                ],
                style=box_style),
            html.Div(
                [
                    html.H3("Average Interest Rate", style=h3_style),
                    html.P("8.02%", style=p_style)
                ],
                style=box_style),
            html.Div(
                [
                    html.H3("Average Loan Amount", style=h3_style),
                    html.P("2,81,876", style=p_style)
                ],
                style=box_style)
        ],
            style={
            "display": "flex",
            "justifyContent": "space-between",
        }
        ),
        html.Div(
            [
                html.H3("Customer Status Distribution", style=h3_style),
                dcc.Graph(figure=plot_pie_chart('customer_status'))
            ],
            style={
                "margin": 'auto',
                "width": "50%",
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '12px',
                'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
                'textAlign': 'center'}),

        html.H1("Trends", style=h1_style),
        html.Div([
            html.Div([
                html.H3("Average Loan over the years", style=h3_style),
                dcc.Graph(figure=plot_line_graph('yoy_trend_average_loan','TotalAdv','Month-Year'))

            ], style=box_style),
            html.Div([
                html.H3("Number of loan disbursed over the years",
                        style=h3_style),
                dcc.Graph(figure=plot_line_graph('yoy_trend_count','Count','Month-Year'))

            ], style=box_style),
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
        }),
        html.H1("Correlation among Numerical features",
                style=h1_style),
        html.Div([
            dcc.Graph(figure=plot_corr('assets/plots/correlation_matrix.csv'))
        ], style={
            "margin": 'auto',
            # "width": "50%",
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
            'textAlign': 'center'}),
        html.H1("Numerical feature Visualization", style=h1_style),
        html.Div([
            html.Label("Column:", style={
                       'color': 'black', 'textAlign': 'left'}),
            dcc.Dropdown(
                id='num-dropdown',
                options=[{'label': file.split(".")[0], 'value': file}
                         for file in num_files],
                value='age.csv',
                placeholder="Select a Column...",
                style={'color': 'black', 'width': '50%',
                       'marginBottom': '20px', 'textAlign': 'left'}
            ),

            dcc.Graph(id='histogram-plot')
        ], style=box_style),
        html.H1("Categorical Column Visualization", style=h1_style),
        html.Div([
            html.Label("Column:", style={
                       'color': 'black', 'textAlign': 'left'}),
            dcc.Dropdown(
                id='cat-dropdown',
                options=[{'label': file.split(".")[0], 'value': file}
                         for file in cat_files],
                value='area.csv',
                placeholder="Select a Column...",
                style={'color': 'black', 'width': '50%',
                       'marginBottom': '20px', 'textAlign': 'left'}
            ),

            dcc.Graph(id='bar-plot'),
        ], style=box_style),
        html.H1("Multivariate Visualization", style=h1_style),
        html.Div([
            html.Label("Column:", style={
                       'color': 'black', 'textAlign': 'left'}),
            dcc.Dropdown(
                id='multi-dropdown',
                options=[{'label': file.split(".")[0], 'value': file}
                         for file in multi_files],
                value='area_wise_customer_status.csv',
                placeholder="Select a Column...",
                style={'color': 'black', 'width': '50%',
                       'marginBottom': '20px', 'textAlign': 'left'}
            ),

            dcc.Graph(id='multi-plot'),
        ], style=box_style)
    ],
    style={'padding': '20px'}
)


# Callback to update histogram based on selected file
@dash.callback(
    Output('histogram-plot', 'figure'),
    Input('num-dropdown', 'value')
)
def update_histogram(selected_file):
    if not selected_file:
        # Return an empty figure if no file is selected
        return go.Figure()

    # Load the selected CSV file
    csv_path = os.path.join(num_directory, selected_file)
    df = pd.read_csv(csv_path)

    fig = px.histogram(df, x=df[df.columns[1]],
                       y=df[df.columns[3]],
                       color_discrete_sequence=px.colors.qualitative.Pastel2
                       )

    # Define the layout
    fig.update_layout(
        # margin={"r": 0, "t": 200, "l": 0, "b": 0},
        bargap=0.01,
        # title=f"Histogram for {selected_file.split('.')[0]}",
        xaxis=dict(title=f"{selected_file.split('.')[0]}"),
        yaxis=dict(title="Count"),
        template='ggplot2',
        # annotations=[
        #     dict(
        #         x=0.5,  # Position at the center of the plot
        #         y=1.5,  # Slightly above the plot
        #         xref="paper",
        #         yref="paper",
        #         text=annotation[selected_file.split('.')[0]],
        #         showarrow=True,  # No arrow pointing to the text
        #         # Customize the font size and color
        #         font=dict(size=14, color="black"),
        #         align="center"  # Center the text
        #     )
        # ]
    )

    return fig


@dash.callback(
    Output('bar-plot', 'figure'),
    Input('cat-dropdown', 'value')
)
def update_bar_graph(selected_file):
    if not selected_file:
        # Return an empty figure if no file is selected
        return go.Figure()

    # Load the selected CSV file
    csv_path = os.path.join(cat_directory, selected_file)
    df = pd.read_csv(csv_path)

    fig = px.histogram(df, x=df[df.columns[0]],
                       y=df[df.columns[1]], color_discrete_sequence=['turquoise'])

    # fig.add_trace(marker_color='#87EBA4')
    # Define the layout
    fig.update_layout(
        bargap=0.1,
        # title=f"Histogram for {df.columns[0]}",
        xaxis=dict(title=f"{df.columns[0]}"),
        yaxis=dict(title="Count"),
        template='ggplot2',

    )
    fig.update_xaxes(categoryorder='total descending')

    return fig


@dash.callback(
    Output('multi-plot', 'figure'),
    Input('multi-dropdown', 'value')
)
def update_bar_graph(selected_file):
    if not selected_file:
        return go.Figure()

    csv_path = os.path.join(multi_directory, selected_file)
    df = pd.read_csv(csv_path)

    df_long = df.melt(id_vars=f'{df.columns[0]}',
                      value_vars=[f'{df.columns[1]}', f'{df.columns[2]}', f'{df.columns[3]}',
                                  f'{df.columns[4]}', f'{df.columns[5]}', f'{df.columns[6]}'],
                      var_name="Category",
                      value_name="Value")

    df_long["Value"] = df_long["Value"].fillna(0)

    fig = px.bar(df_long,
                 x=f'{df.columns[0]}',
                 y="Value",
                 color="Category",
                #  title=f'Stacked Bar Plot for {df.columns[0]}',
                 text_auto=True,
                 color_discrete_sequence=['#F6F926','#16FF32','#0DF9FF','#1CBE4F','#FEAF16','#FED4C4']
                 )

    return fig
