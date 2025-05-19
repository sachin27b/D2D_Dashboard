'''
Things to do in this page:
What if analysis
1. enter values of each feature
    a. numerical values should be entered
    b. categorical values should be selected from dropdown
2. transform these values
3. load model (pd & lgd)
4. pass to data to both model
5. show prediction both pd & lgd
6. Calculate ead and d2d
7. show firb table'''

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
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
import json
import dash_table
import time
from dash.exceptions import PreventUpdate

from scipy.stats import norm


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

COLOR_PALETTE2 = {
    'primary_dark': '#0A1128',
    'primary_mid': '#1B2845',
    'accent_blue': '#1CB5E0',
    'accent_green': '#5CDB95',
    'text_light': '#F7F9FB',
    'text_subtle': '#B0BEC5'
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
 'RepayMode',
 'idyr',
 'NoOfInstall',
 'Cust_AssetClass',
 'BaselCatMark']


numerical_cols = ['Age',
 'CAD',
 'CADU',
 'UsedRv',
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
 'PriCashSecurity',
 'CollCashSecurity',
 'CollNonCRMSecurity',
 'OrgCostOfPlantMech',
 'Principal',
 'ResidualExposure',
 'LimitSanctioned',
 'Balance',
 'FstInstallDt',
 'ProcessingDt',
 'NpaDt',
 'DtofFirstDisb']

with open('assets/column_mapping.json','rb') as file:
    column_mapping = json.load(file)

with open("assets/encoding_list.pkl", "rb") as file:
    data = pickle.load(file)
    
sample = pd.read_csv("assets/sample.csv")
sample = sample.iloc[:1]

spark = SparkSession.builder \
    .appName("what_if") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "10g") \
    .master("local[*]") \
    .getOrCreate()

best_cl_helper_obj = helper.Helper()
best_cl_artifact_obj = get_artifact.DownloadArtifact(best_cl_helper_obj)


best_cl_model_path = best_cl_artifact_obj.get_model()
classifier = mlflow.spark.load_model(best_cl_model_path)

best_reg_helper_obj = helper.Helper('xgb_regressor')
best_reg_artifact_obj = get_artifact.DownloadArtifact(best_reg_helper_obj)

best_reg_model_path = best_reg_artifact_obj.get_model()
regressor = mlflow.spark.load_model(best_reg_model_path)


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



def get_encodings(df, encoding_data):

    encoding = {}
    for col in df:
        # print(col,value)
        for index, encoding_dict in enumerate(encoding_data):
            # print(index,encoding_dict)
            if col in encoding_dict:
                # print(col)
                for key, val in encoding_dict[col].items():
                    # print(key,val)
                    if val == df[col].values:
                        mean_encoding_key = f"{col}_mean_encoding"
                        encoding[mean_encoding_key] = encoding_data[index][mean_encoding_key][key]                   

    return encoding

def get_date_encoding(df):
    date_columns = ['FstInstallDt', 'ProcessingDt', 'NpaDt', 'DtofFirstDisb']

    for col in date_columns:
        # Convert to datetime first (if not already in datetime format)
        df[col] = pd.to_datetime(df[col])
        # Convert to Unix timestamp (seconds since epoch)
        df[f'{col}_unix'] = df[col].apply(lambda x: int(x.timestamp()))
        
    return df

def standard_scaler(df):
    with open('assets/std_values.json', 'r') as file:
        std_values = json.load(file)
    for col_name, std in std_values.items():
        df[f'{col_name}_scaled'] = df[col_name] / std
    return df


def calculate_ead(rawdata):

    # Function to replace NA values with zero
    def na_to_zero(series):
        return series.fillna(0)

    # Loan Tenure Calculation
    loan_tenure = np.zeros(len(rawdata))
    conditions = [
        (rawdata["RepayMode"] == 1) | (rawdata["RepayMode"] == 10),
        (rawdata["RepayMode"] == 2) | (rawdata["RepayMode"] == 11),
        (rawdata["RepayMode"] == 3) | (rawdata["RepayMode"] == 12),
        (rawdata["RepayMode"] == 4) | (rawdata["RepayMode"] == 13)
    ]
    divisors = [12, 4, 2, 1]
    for cond, divisor in zip(conditions, divisors):
        loan_tenure[cond] = rawdata.loc[cond, "NoOfInstall"] / divisor

    # Special case for repayment mode 5
    loantendays5 = pd.to_datetime(rawdata["FstInstallDt"]) - pd.to_datetime(rawdata["DtofFirstDisb"])
    loan_tenure[rawdata["RepayMode"] == 5] = loantendays5.dt.days[rawdata["RepayMode"] == 5] / 365

    # Adjust tenure to whole numbers and cap at 30
    loan_tenure = np.ceil(loan_tenure)
    loan_tenure[loan_tenure > 30] = 30
    rawdata["LoanTenure"] = loan_tenure

    # Loan Age Calculation
    npa_dt = pd.to_datetime(rawdata["NpaDt"], errors="coerce")
    disbursement_dt = pd.to_datetime(rawdata["DtofFirstDisb"])
    npa_dt = npa_dt.fillna(pd.to_datetime(rawdata["idyr"].astype(str) + "-03-31"))
    days_at_def = (npa_dt - disbursement_dt).dt.days
    loan_age = days_at_def / 365
    rawdata["LoanAge"] = loan_age

    # Residual Tenure Calculation
    res_tenure = loan_tenure - loan_age
    res_tenure[res_tenure < 0] = 0
    res_tenure[res_tenure > 30] = 30
    rawdata["ResTenure"] = res_tenure

    # EAD Calculation
    principal = na_to_zero(rawdata["Principal"])
    limit_sanctioned = na_to_zero(rawdata["LimitSanctioned"])
    rawdata["EAD"] = np.where(
        rawdata["ResTenure"] <= 1,
        principal + 0.2 * (limit_sanctioned - principal),
        principal + 0.5 * (limit_sanctioned - principal)
    )

    # Residual Exposure Calculation
    rawdata["ResExp"] = rawdata["ResidualExposure"] / rawdata["Balance"]

    return rawdata


def calculate_d2d(m):

    # Extract inputs
    pd_values = m['probability'].values[0][1]
    rr = m["regressor_prediction"].fillna(0.75).clip(0, 1).values

    # Transformation of PD
    tpd = pd_values**0.1
    # if len(tpd) > 1:
    #     ztpd = (tpd - np.mean(tpd)) / np.std(tpd)
    #     ztpd = np.clip(ztpd, -4, 4)
    #     ztpd = ztpd / (ztpd.max() - ztpd.min())
    #     ztpd = (ztpd - ztpd.min() - 0.5) * 8
    # else:
    ztpd = (tpd - 0.67) / 0.27
    ztpd = np.clip(ztpd, -4, 4) / 4
    ztpd = ztpd * 8

    d2d = np.round(4 - ztpd, 1)

    ead = m["EAD"].values
    lgd = 1 - rr
    resexp = m["ResExp"].values

    r = 0.03 * ((1 - np.exp(-35 * pd_values)) / (1 - np.exp(-35))) + \
        0.16 * ((np.exp(-35 * pd_values) - np.exp(-35)) / (1 - np.exp(-35)))

    x = (norm.ppf(pd_values) + (r**0.5) * norm.ppf(0.9999)) / (1 - r**0.5)
    k = lgd * norm.cdf(x) - lgd * pd_values
    rwa = k * 12.5 * ead
    rwa = np.maximum(rwa, 0)
    riskwt = 100 * rwa / ead

    # Residual Exposure Provisions
    resexp[np.isinf(resexp)] = 1
    resexp[np.isnan(resexp)] = 0
    prov = resexp * norm.cdf(ztpd, 0, 1.5) / 10

    # AIRB DataFrame
    airb = pd.DataFrame({
        # "ACID": m["MapKey"],
        "D2D": d2d,
        "PD": np.round(pd_values, 4),
        "EAD": np.round(ead, 0),
        "LGD": np.round(lgd, 4),
        "K": np.round(k, 4),
        "RWA": np.round(rwa, 0),
        "Risk Wt": np.round(riskwt, 2),
        "Res Exp": np.round(resexp, 0),
        "Provision": np.round(prov, 0)
    })
    
    return airb

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
                html.Label(column_mapping['categorical_cols'][categorical_cols[i]],
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
                html.Label(column_mapping['numerical_cols'][num_col],
                           style=h3_style),
                dcc.Input(
                    id=f'input-{num_col}',
                    value=sample[num_col][0],
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
    dcc.Loading(
        type="circle",
        children=html.Div(id='output-container', style={'marginTop': '20px'})
    ),
    # html.Div(id='output-container', style={'marginTop': '20px'})
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

        cat_vals = {col: val for col, val in zip(
            categorical_cols, dropdown_values)}
        num_vals = {num_col: val for num_col,
                          val in zip(numerical_cols, input_values)}
        merged_vals = {**num_vals,**cat_vals}

        df = pd.DataFrame([merged_vals])
        df_copy = df.copy()
        
        encoding_df = pd.DataFrame([get_encodings(df,data)])
        df = pd.concat([df, encoding_df], axis=1)
        
        df = get_date_encoding(df)
        
        df = standard_scaler(df)
        StandardScaler_columns = [col for col in df.columns if 'scaled' in col]
        mean_encoded_columns = [col for col in df.columns if 'encoding' in col]


        data_df = spark.createDataFrame(df[StandardScaler_columns+mean_encoded_columns])
   
        
        final_assembler = VectorAssembler(inputCols=StandardScaler_columns+mean_encoded_columns, outputCol="features")
        what_if_data = final_assembler.transform(data_df)
    
        classifier_pred = classifier.transform(what_if_data)
    
        regressor_pred = regressor.transform(what_if_data)

        # print(classifier_pred.show())
        # print(regressor_pred)
        # classifier_pred.show()
        # regressor_pred.show()


        classifier_pred = classifier_pred.toPandas()
        regressor_pred = regressor_pred.toPandas()
        
        classifier_pred.rename(columns={'prediction':'classifier_prediction'},inplace=True)
        regressor_pred.rename(columns={'prediction':'regressor_prediction'},inplace=True)
        
        final_df = pd.concat([classifier_pred,regressor_pred['regressor_prediction']],axis=1)
        final_df['regressor_prediction'] = np.expm1(final_df['regressor_prediction'])
        
        final_df = pd.concat([df_copy,final_df],axis=1)
        
        final_df = calculate_ead(final_df)
        
        airb = calculate_d2d(final_df)
        
        airb_table = html.Table([
                            html.Thead(
                                html.Tr([html.Th(col, style={'padding': '10px', 'backgroundColor': COLOR_PALETTE2['accent_blue'], 'color': 'white'}) for col in airb.columns])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(airb.iloc[row][col], style={'padding': '8px', 'border': '1px solid #ddd','color': 'darkblue'}) 
                                    for col in airb.columns
                                ]) for row in range(len(airb))
                            ])
                        ], style={
                            'width': '100%', 
                            'borderCollapse': 'collapse', 
                            'marginBottom': '20px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                        })
        return dcc.Loading(
    type="circle",
    children=html.Div([
        html.Div([
            html.H3(f'Predicted Class: {round(final_df["classifier_prediction"].values[0])}', style={'color': '#00EA64'}),
            html.H4(f'Predicted Recovery Rate: {round(final_df["regressor_prediction"].values[0],2)}', style={'color': '#00EA64'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # DataTable to display the DataFrame
        html.Div([
            html.H3(
                'A-IRB', 
                style={
                    'color': COLOR_PALETTE2['primary_mid'],
                    'textAlign': 'center',
                    'marginBottom': '15px'
                }
            ),
            airb_table
        ], style={
            'maxHeight': '250px',
            'overflowY': 'auto',
            'border': '1px solid #ddd',
            'marginTop': '30px',
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 6px 12px rgba(0,0,0,0.1)'
        })
    ])
)
