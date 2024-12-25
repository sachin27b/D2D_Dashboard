'''
Things to do in this page:
What if analysis
1. enter values of each feature
    a. numerical values should be entered
    b. categorical values should be selected from dropdown
2. transform these values
3. load model
4. pass to data to model'''

import dash
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from components import helper, get_artifact
from pyspark.sql import Row
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import pickle
from dotenv import load_dotenv
import os
import mlflow

load_dotenv(dotenv_path='../config/.env')
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

COLOR_PALETTE = {
    'background_dark': '#121212',
    'background_mid': '#1E1E1E',
    'primary_accent': '#BB86FC',
    'secondary_accent': '#03DAC6',
    'text_primary': '#FFFFFF',
    'text_secondary': '#B0BEC5'
}

dash.register_page(__name__, path='/what-if')

categorical_cols = ['GL',
                    'Area',
                    'Caste',
                    'Scheme',
                    'Activity',
                    'District',
                    'Religion',
                    'SubSector',
                    'Occupation',
                    'FORMATGROUP',
                    'Constitution',
                    'FACILITYTYPE',
                    'CUSTOMERSTATUS',
                    'FACILITIESSTATUS',
                    ]


numerical_cols = ['Age',
                  'CAD',
                  'CADU',
                  'UsedRv',
                  'Balance',
                  'Product',
                  'TotOsNF',
                  'AdhocAmt',
                  'InttRate',
                  'InttType',
                  'TotalAdv',
                  'AppGovGur',
                  'CURQTRINT',
                  'GLProduct',
                  'PRVQTRINT',
                  'RvPrimary',
                  'Unsecured',
                  'TotLimitNF',
                  'CoverGovGur',
                  'DepValPlant',
                  'ACCTotalProv',
                  'BaselCatMark',
                  'CURRENTLIMIT',
                  'CurQtrCredit',
                  'DRAWINGPOWER',
                  'PREQTRCREDIT',
                  'UnAdjSubSidy',
                  'UnappliedInt',
                  'ProvUnsecured',
                  'TotalWriteOff',
                  'AppropriatedRv',
                  'OrgCostOfEquip',
                  'TotLimitFunded',
                  'Cust_AssetClass',
                  'LimitSanctioned',
                  'PriCashSecurity',
                  'CollCashSecurity',
                  'CollNonCRMSecurity',
                  'OrgCostOfPlantMech']


file_path = "assets/encoding_list.pkl"

with open(file_path, "rb") as file:
    data = pickle.load(file)

spark = SparkSession.builder \
    .appName("what_if") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "10g") \
    .master("local[*]") \
    .getOrCreate()

helper_obj = helper.Helper()
artifact_obj = get_artifact.DownloadArtifact(helper_obj)


model_path = artifact_obj.get_model()
loaded_model = mlflow.spark.load_model(model_path)

box_style = {
    'backgroundColor': 'white',
    'padding': '20px',
    'borderRadius': '12px',
                    'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
                    'margin': '10px',
                    'flex': '1',
                    'textAlign': 'center'
}

h3_style = {
    'color': '#1CB5E0',
    'fontSize': '18px',
    'fontWeight': 700,
    'margin': '0',
    'overflowWrap': 'break-word'
}

h1_style = {
    'color': '#1B2845',
    'textAlign': 'left',
    'marginBottom': '20px',
    'fontWeight': '600',
    'fontSize': '3rem',
    'textShadow': '0 0 10px rgba(255, 255, 255, 0.7), 0 0 20px ' + '#1B2845'}


def find_encodings(categorical_columns, data):

    encoding = {}

    for col, value in categorical_columns.items():
        for index, encoding_dict in enumerate(data):
            if col in encoding_dict:
                for key, val in encoding_dict[col].items():
                    if val == value:
                        mean_encoding_key = f"{col}_mean_encoding"
                        encoding[col] = data[index][mean_encoding_key][key]

    return encoding


def transform_numerical_columns(numerical_columns, df):

    assembler = VectorAssembler(inputCols=list(
        numerical_columns.keys()), outputCol="features")
    assembled_df = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    transformed_values = {}
    scaled_data = scaled_df.select("scaledFeatures").collect()
    for i, col in enumerate(numerical_columns.keys()):
        transformed_values[col] = scaled_data[0]["scaledFeatures"][i]

    return transformed_values


dropdown_options = [
    {col: list(data_item[col].values())}
    for col in categorical_cols
    for data_item in data
    if col in data_item
]

layout = html.Div([
    html.H1('What-If Analysis',
            style=h1_style),

    html.Div([
        html.Div([
            html.Div([
                html.Label(categorical_cols[i],
                           style=h3_style),
                dcc.Dropdown(
                    options=dropdown_options[i][categorical_cols[i]],
                    value=dropdown_options[i][categorical_cols[i]][0],
                    id=f'dropdown-{categorical_cols[i]}',
                    style={
                        'width': '100%',
                        'color': 'black',
                        # 'padding': '8px',
                        'border': f'1px solid {COLOR_PALETTE["primary_accent"]}',
                        'borderRadius': '5px'
                    },
                    clearable=True
                )
            ], style={'marginBottom': '15px'})
            for i in range(len(dropdown_options))
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '15px'}),

        html.Div([
            html.Div([
                html.Label(num_col,
                           style=h3_style),
                dcc.Input(
                    id=f'input-{num_col}',
                    value=0,
                    type='number',
                    placeholder=f'Enter {num_col}',
                    style={
                        'width': '100%',
                        # 'padding': '8px',
                        'border': f'2px solid {COLOR_PALETTE["primary_accent"]}',
                        'borderRadius': '5px'
                    }
                )
            ], style={'marginBottom': '15px'})
            for num_col in numerical_cols
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '15px'}),
    ]),

    html.Div([
        html.Button('Submit',
                    id='submit-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': COLOR_PALETTE['secondary_accent'],
                        'color': COLOR_PALETTE['text_primary'],
                        'border': 'none',
                        # 'padding': '10px 10px',
                        'borderRadius': '5px',
                        'fontSize': '16px',
                        'fontWeight': 'bold',
                        'cursor': 'pointer',                    }
                    )
    ], style={'textAlign': 'center', 'marginTop': '5px'}),

    html.Div(id='output-container', style={'marginTop': '20px'})
], style=box_style)


@dash.callback(
    Output('output-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State(f'dropdown-{col}', 'value') for col in categorical_cols] +
    [State(f'input-{num_col}', 'value') for num_col in numerical_cols]
)
def handle_values(n_clicks, *values):
    if n_clicks > 0:
        # Existing logic for processing values
        dropdown_values = values[:len(categorical_cols)]
        input_values = values[len(categorical_cols):]

        selected_values = {col: val for col, val in zip(
            categorical_cols, dropdown_values)}
        entered_values = {num_col: val for num_col,
                          val in zip(numerical_cols, input_values)}

        combined_values = {'Dropdown Values': selected_values,
                           'Entered Values': entered_values}

        # print(selected_values)
        # print(entered_values)

        categorical_encodings = find_encodings(selected_values, data)
        num_spark_df = spark.createDataFrame([entered_values])
        numerical_encodings = transform_numerical_columns(
            entered_values, num_spark_df)
        final_data = {**categorical_encodings, **numerical_encodings}
        final_data_df = {key: float(value) if isinstance(
            value, (int, float)) else value for key, value in final_data.items()}

        final_df = spark.createDataFrame([final_data_df])

        final_features = numerical_cols + categorical_cols
        final_df = final_df.select([col for col in final_features])

        final_assembler = VectorAssembler(
            inputCols=final_features, outputCol="features")
        what_if_data = final_assembler.transform(final_df)
        pred = loaded_model.transform(what_if_data)
        probability_list = pred.select(
            "probability").rdd.flatMap(lambda x: x).collect()

        return html.Div([
            html.H3('Prediction Results', style={
                    'color': COLOR_PALETTE['secondary_accent']}),
            html.Pre(f'Predicted probability of Class 0: {round(probability_list[0][0],3)}', style={
                'backgroundColor': COLOR_PALETTE['background_mid'],
                'padding': '15px',
                'borderRadius': '5px',
                'color': COLOR_PALETTE['text_secondary']
            }),
            html.Pre(f'Predicted probability of Class 1: {round(probability_list[0][1],3)}', style={
                'backgroundColor': COLOR_PALETTE['background_mid'],
                'padding': '15px',
                'borderRadius': '5px',
                'color': COLOR_PALETTE['text_secondary']
            })
        ])
