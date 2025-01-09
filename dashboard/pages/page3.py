'''
TODO PD
TODO LGD
1. main metric
2. other metrics
3. feature importance
4. '''
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
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

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

dash.register_page(__name__, path='/model')

pd_dropdown_options = [
    {'label': 'XGBoost', 'value': 'xgboost'},
    {'label': 'Logistic Regression', 'value': 'logistic_regression'},
    {'label': 'Decision Tree', 'value': 'dt_classifier'},
    {'label': 'Gradient Boosted Tree', 'value': 'gbt_classifier'}
]
lgd_dropdown_options = [
    {'label': 'XGB Regressor', 'value': 'xgb_regressor'},
    {'label': 'Decision tree', 'value': 'decision_tree_regressor'},
    {'label': 'Gradient Boosted tree', 'value': 'gradient_boosted_tree_regressor'},
    {'label': 'Random Forest', 'value': 'random_forest_regressor'},
    {'label': 'Linear Regression', 'value': 'linear_regression'}
]
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

tabs_styles = {
    'height': '44px',
    'margin': '20px 0',
    'border-radius': '8px',
    'background-color': '#f8f9fa',
    'alignItems': 'center'
}

tab_style = {
    'borderBottom': '1px solid #dee2e6',
    'padding': '12px 16px',
    'fontWeight': '500',
    'color': '#495057',
    'backgroundColor': '#ffffff',
    'borderRadius': '8px 8px 0 0',
    'margin-right': '2px',
    'transition': 'all 0.3s ease-in-out',
    'cursor': 'pointer',
    'fontSize': '18px',
    'display': 'flex',
    'alignItems': 'center'
}

tab_selected_style = {
    'borderTop': '2px solid #4361ee',
    'borderLeft': '1px solid #dee2e6',
    'borderRight': '1px solid #dee2e6',
    'borderBottom': 'none',
    'backgroundColor': '#ffffff',
    'color': '#4361ee',
    'padding': '12px 16px',
    'fontWeight': '600',
    'borderRadius': '8px 8px 0 0',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.04)',
    'position': 'relative',
    'fontSize': '18px',
    'alignItems': 'center'
}

pd_model_list = ['xgboost','dt_classifier','logistic_regression','gbt_classifier']
model_color = ['red', 'blue', 'green', 'black']

lgd_model_list = ['xgb_regressor','decision_tree_regressor','gradient_boosted_tree_regressor','random_forest_regressor','linear_regression']

def plot_guage(model_accuracy):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=model_accuracy * 100,
        title={'text': "Model Accuracy (%)"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "teal"},
                {'range': [50, 85], 'color': "royalblue"},
                {'range': [85, 100], 'color': "cyan"}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': model_accuracy * 100
            }
        }
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def plot_radar(model_metrics):
    metrics = ['F1 Score', 'FPR', 'TPR', 'Precision', 'ROC AUC']
    values = [
        round(model_metrics['f1_score'], 3),
        round(model_metrics['false_positive_rate'], 3),
        round(model_metrics['true_positive_rate'], 3),
        round(model_metrics['precision'], 3),
        round(model_metrics['auc'], 3)
    ]

    values += [values[0]]
    metrics += [metrics[0]]  

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name='Model Metrics'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        # title="Radar Plot of Model Metrics",
        showlegend=True
    )

    return fig

def plot_lgd_radar(model_metrics):
    metrics = ['RMSE', 'R Square', 'MAE']
    values = [
        round(model_metrics['rmse'], 3),
        round(model_metrics['r2'], 3),
        round(model_metrics['mae'], 3),
    ]

    values += [values[0]]
    metrics += [metrics[0]]  

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name='Model Metrics'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
    )

    return fig


def plot_confusion_matrix(path):

    with open(path, 'r') as file:
        confusion_matrix = json.load(file)

    # Create the 2x2 matrix
    matrix = [
        [confusion_matrix["True Negative"], confusion_matrix["False Positive"]],
        [confusion_matrix["False Negative"], confusion_matrix["True Positive"]]
    ]

    # Labels for the axes
    labels = ["Negative", "Positive"]

    # Plot the confusion matrix
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,  # Predicted labels
            y=labels,  # Actual labels
            colorscale="Blues",
            showscale=True
        )
    )

    # Add annotations
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=str(matrix[i][j]),
                showarrow=False,
                font=dict(color="black", size=14)
            )

    # Update layout
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        template="plotly_white"
    )

    return fig


def plot_roc(path):

    with open(path, 'r') as file:
        roc_data = json.load(file)

    if len(roc_data['fpr']) > 3000:
        step = 300
        fpr_reduced = roc_data["fpr"][::step]
        tpr_reduced = roc_data["tpr"][::step]
    else:
        fpr_reduced = roc_data["fpr"]
        tpr_reduced = roc_data["tpr"]
    # Plot the ROC curve
    fig = go.Figure()

    # Add the line for ROC
    fig.add_trace(
        go.Scatter(
            x=fpr_reduced,
            y=tpr_reduced,
            mode="lines+markers",
            name="ROC Curve",
            line=dict(color="blue", width=2),
            marker=dict(size=6)
        )
    )

    # Add layout details
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        xaxis=dict(range=[0, 1], showgrid=True),
        yaxis=dict(range=[0, 1], showgrid=True),
        template="plotly_white",
        showlegend=True
    )

    return fig


def plot_feat_importance(path):
    with open(path, 'r') as file:
        feature_importance_dict = json.load(file)
    # Sort the dictionary by importance values in descending order

    if "LogisticRegression" in path:
        feature_importance_dict['coefficients'] = {
            key: abs(value)
            for key, value in feature_importance_dict['coefficients'].items()
        }
        sorted_features = sorted(
            feature_importance_dict['coefficients'].items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    top_10_features = sorted_features[:10]
    top_10_feature_names = [item[0] for item in top_10_features]
    top_10_importance_values = [item[1] for item in top_10_features]
    top_10_feature_names = top_10_feature_names[::-1]
    top_10_importance_values = top_10_importance_values[::-1]

    # Create the horizontal bar chart using Plotly
    fig = go.Figure(go.Bar(
        x=top_10_importance_values,
        y=top_10_feature_names,
        orientation='h',  
        marker=dict(color='cyan')
    ))

    # Customize the layout
    fig.update_layout(
        title="Feature Importances(Top 10)",
        xaxis_title="Importance",
        yaxis_title="Features",
        template="plotly_dark",
        showlegend=False
    )

    return fig

def plot_pd_parallel(pd_model_list,model_color):
    metrics_data = {}
    model = []
    f1 = []
    accuracy = []
    fpr = []
    tpr = []
    auc = []
    precision = []

    for name in pd_model_list:
        helper_obj1 = helper.Helper(name)
        model_metrics = helper_obj1.get_metrics()
        model.append(helper_obj1.get_run().data.tags['mlflow.runName'])
        accuracy.append(model_metrics['accuracy'])
        f1.append(model_metrics['f1_score'])
        auc.append(model_metrics['auc'])
        fpr.append(model_metrics['false_positive_rate'])
        tpr.append(model_metrics['true_positive_rate'])
        precision.append(model_metrics['precision'])
        
    metrics_data = {
        'Model': model,
        'Accuracy': accuracy,
        'F1': f1,
        'AUC': auc,
        'FPR': fpr,
        'TPR': tpr,
        'Precision': precision,
    }
    df = pd.DataFrame(metrics_data)

    # Map model names to numeric values
    df['Model_Num'] = df['Model'].astype('category').cat.codes
    fig = px.parallel_coordinates(
        df,
        dimensions=["F1", "Accuracy", "FPR", "TPR", "Precision"], 
        color="Model_Num",
        labels={
            "F1": "F1 Score",
            "Accuracy": "Accuracy",
            "FPR": "False Positive Rate",
            "TPR": "True Positive Rate",
            "Precision": "Precision",
            "Model_Num": "Model"
        },
        color_continuous_scale=model_color  
    )

    return fig

def plot_lgd_parallel(lgd_model_list,model_color):
    metrics_data = {}
    model = []
    rmse = []
    mse = []
    r2 = []
    mae = []

    for name in lgd_model_list:
        helper_obj1 = helper.Helper(name)
        model_metrics = helper_obj1.get_metrics()
        model.append(helper_obj1.get_run().data.tags['mlflow.runName'])
        rmse.append(model_metrics['rmse'])
        mse.append(model_metrics['mse'])
        r2.append(model_metrics['r2'])
        mae.append(model_metrics['mae'])
        
    metrics_data = {
        'Model': model,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2,
        'MAE': mae,
    }
    df = pd.DataFrame(metrics_data)

    # Map model names to numeric values
    df['Model_Num'] = df['Model'].astype('category').cat.codes
    fig = px.parallel_coordinates(
        df,
        dimensions=["RMSE", "MSE", "R2", "MAE"], 
        color="Model_Num",
        labels={
            "RMSE": "RMSE",
            "MSE": "MSE",
            "R2": "R2",
            "MAE": "MAE",
            "Model_Num": "Model"
        },
        color_continuous_scale=model_color  
    )

    return fig


import json
import seaborn as sns
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(data_path, output_path):
    # Load data from the JSON file
    with open(data_path, 'rb') as file:
        data = json.load(file)
    
    # Create a scatter plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=data["actual"], 
        y=data["predicted"], 
        color='blue', 
        s=80, 
        label='Data Points'
    )
    
    # Add the diagonal line (perfect prediction line)
    min_val = min(min(data["actual"]), min(data["predicted"]))
    max_val = max(max(data["actual"]), max(data["predicted"]))
    plt.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        color='green', 
        linestyle='--', 
        label='Perfect Prediction'
    )
    
    # Add titles and labels
    plt.title("Actual vs Predicted", fontsize=16)
    plt.xlabel("Actual Values", fontsize=14)
    plt.ylabel("Predicted Values", fontsize=14)
    plt.legend(title="Legend", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot as a PNG file
    plt.savefig(output_path, bbox_inches='tight')
    # plt.close()
    
    return output_path


theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

layout = html.Div(
    [
        dcc.Tabs([
            dcc.Tab(label='PD',children = [
        html.Div(
            [
                html.H3("PD Model Comparison", style=h3_style),
                dcc.Graph(figure=plot_pd_parallel(pd_model_list,model_color)),
                html.Div(
                            [
                                html.Div("Legend", style=h3_style),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(style={"width": "20px", "height": "20px", "backgroundColor": f'{m_color}', "display": "inline-block", "marginRight": "10px"}),
                                                html.Span(f'{m_list}', style={'color':'black'})
                                            ],
                                            style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}
                                        ) for m_color, m_list in zip(model_color,pd_model_list)
                                    ],
                                    style={"border": "1px solid #ccc", "padding": "15px", "borderRadius": "5px", "maxWidth": "300px"}
                                )
                            ],
                            style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start", "padding": "20px"}
                        )
            ],
            style=box_style),

        html.H1("PD Models", style=h1_style),
            dcc.Dropdown(
                id='pdmodel-dropdown',
                options=pd_dropdown_options,
                placeholder='Select a model...',
                style={
                    'width': '50%',
                    'color': 'black',
                }
            ),
            dcc.Loading(
        type="circle",
        children=html.Div(
                id='pdmetrics-content',
            ))
            ],style=tab_style,selected_style=tab_selected_style),
            # 
            dcc.Tab(label='LGD',children=[
                html.Div(
            [
                html.H3("LGD Model Comparison", style=h3_style),
                dcc.Graph(figure=plot_lgd_parallel(lgd_model_list,model_color)),
                html.Div(
                            [
                                html.Div("Legend", style=h3_style),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(style={"width": "20px", "height": "20px", "backgroundColor": f'{m_color}', "display": "inline-block", "marginRight": "10px"}),
                                                html.Span(f'{m_list}', style={'color':'black'})
                                            ],
                                            style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}
                                        ) for m_color, m_list in zip(model_color,lgd_model_list)
                                    ],
                                    style={"border": "1px solid #ccc", "padding": "15px", "borderRadius": "5px", "maxWidth": "300px"}
                                )
                            ],
                            style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start", "padding": "20px"}
                        )
            ],
            style=box_style),

        html.H1("LGD Models", style=h1_style),
            dcc.Dropdown(
                id='lgdmodel-dropdown',
                options=lgd_dropdown_options,
                placeholder='Select a model...',
                style={
                    'width': '50%',
                    'color': 'black',
                }
            ),
            dcc.Loading(
        type="circle",
        children=html.Div(
                id='lgdmetrics-content',
            ))],style=tab_style,selected_style=tab_selected_style)
        ])
    ])


@dash.callback(
    Output('pdmetrics-content', 'children'),
    Input('pdmodel-dropdown', 'value'),
)
def update_pdmetrics(selected_pdmodel):
    # print(selected_pdmodel)
    if not selected_pdmodel:
        return html.Div(
        )
    # shutil.rmtree('assets/confusion_matrix')
    # shutil.rmtree('assets/roc_curve')

    helper_obj = helper.Helper(selected_pdmodel)
    artifact_obj = get_artifact.DownloadArtifact(helper_obj)
    conf_src = artifact_obj.get_confusion_matrix()
    roc_src = artifact_obj.get_auc()
    feat_importance_src = artifact_obj.get_feat_importances()

    model_metrics = helper_obj.get_metrics()
    
    if not model_metrics:
        return html.Div(
            "Metrics not available for the selected model.",
        )

    metrics_layout = html.Div(
        [
            html.H1(
                f"Metrics for {selected_pdmodel.replace('_', ' ').title()}",
                style=h1_style
            ),

            html.Div([
                html.Div([
                    html.H3("Accuracy", style=h3_style),
                    dcc.Graph(figure=plot_guage(
                        round(model_metrics['accuracy'], 3)))

                ], style=box_style),
                html.Div([
                    html.H3("Key metrics",
                            style=h3_style),
                    dcc.Graph(figure=plot_radar(model_metrics))

                ], style=box_style),
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
            }),
            html.Div(
                [
                    html.H3("Feature Importance", style=h3_style),
                    dcc.Graph(figure=plot_feat_importance(feat_importance_src))
                ],
                style={
                    "margin": 'auto',
                    "width": "70%",
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '12px',
                    'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
                    'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H3("Confusion Matrix", style=h3_style),
                    dcc.Graph(figure=plot_confusion_matrix(conf_src))
                ], style=box_style),
                html.Div([
                    html.H3("ROC Curve",
                            style=h3_style),
                    dcc.Graph(figure=plot_roc(roc_src))

                ], style=box_style),
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
            })
        ]
    )
    return dcc.Loading(
    type="circle",
    children=html.Div([metrics_layout]))


@dash.callback(
    Output('lgdmetrics-content', 'children'),
    Input('lgdmodel-dropdown', 'value'),
)

def update_lgdmetrics(selected_lgdmodel):
    if not selected_lgdmodel:
        return html.Div(
        )
    helper_obj2 = helper.Helper(selected_lgdmodel)
    model_metrics = helper_obj2.get_metrics()
    artifact_obj = get_artifact.DownloadArtifact(helper_obj2)
    actual_vs_predicted_src = artifact_obj.get_actual_vs_predicted()
    # if not model_metrics:
    #     return html.Div(
    #         "Metrics not available for the selected model.",
    #     )
    metrics_layout = html.Div([
                        html.Div([
                                html.Div([
                                    html.H3("MSE", style=h3_style),
                                    daq.DarkThemeProvider(
                                            theme=theme,
                                            children=html.Div([
                                                # LED Display
                                                daq.LEDDisplay(
                                                    value=round(model_metrics['mse'],3),
                                                    color=theme['primary'],
                                                    id='darktheme-daq-leddisplay',
                                                    className='dark-theme-control'
                                                )]))

                                ], style=box_style),
                                html.Div([
                                    html.H3("Key metrics",
                                            style=h3_style),
                                    dcc.Graph(figure=plot_lgd_radar(model_metrics))

                                ], style=box_style),
                            ], style={
                                "display": "flex",
                                "justifyContent": "space-between",
                            }),
                html.Div(
                [
                    html.H3("Actual Vs Predicted", style=h3_style),
                    # plot_actual_vs_predicted(actual_vs_predicted_src,f'/artifacts/actual_vs_predicted/{selected_lgdmodel}.png'),
                    html.Img(src=plot_actual_vs_predicted(actual_vs_predicted_src,f'assets/actual_vs_predicted_{selected_lgdmodel}.png'),
                     style={
                     'width': '90%', 'display': 'block', 'margin': '0 auto',
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '12px',
                    'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
                    # 'margin': '10px',
                    'flex': '1',
                    'textAlign': 'center'})
                ],
                style={
                    "margin": 'auto',
                    "width": "70%",
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '12px',
                    'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
                    'textAlign': 'center'})
                        ])
    return dcc.Loading(
    type="circle",
    children=html.Div([metrics_layout]))