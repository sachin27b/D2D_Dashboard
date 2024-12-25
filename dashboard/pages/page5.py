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

dash.register_page(__name__, path='/backtesting')

pd_dropdown_options = [
    {'label': 'XGBoost', 'value': 'xgboost'},
    {'label': 'Logistic Regression', 'value': 'logistic_regression'},
    {'label': 'Decision Tree', 'value': 'dt_classifier'},
    {'label': 'Gradient Boosted Tree', 'value': 'gbt_classifier'}
]
lgd_dropdown_options = [
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

lgd_model_list = ['decision_tree_regressor','gradient_boosted_tree_regressor','random_forest_regressor','linear_regression']

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
    metrics = ['RMSE', 'R', 'Adjusted R']
    values = [
        round(model_metrics['rmse'], 3),
        round(model_metrics['r2'], 3),
        round(model_metrics['adjusted_r2'], 3),
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
        title="Radar Plot of Model Metrics",
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


helper_obj = helper.Helper('backtesting')
artifact_obj = get_artifact.DownloadArtifact(helper_obj)
conf_src = artifact_obj.get_confusion_matrix()
model_metrics = helper_obj.get_metrics()


layout = html.Div(
        [
            html.H1(
                f"Backtesting Results",
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
            html.Div([
                html.Div([
                    html.H3("Confusion Matrix", style=h3_style),
                    dcc.Graph(figure=plot_confusion_matrix(conf_src))
                ], style=box_style),
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
            })
        ]
    )
