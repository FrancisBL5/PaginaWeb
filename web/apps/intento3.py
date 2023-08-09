from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import networkx as nx
import base64, io
from apps import hassan as hs
from apps import modelos as models

from app import app

#app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
#app.title = 'Página principal'

"""df = pd.DataFrame()
df_sample_train = pd.DataFrame()
df_sample_test = pd.DataFrame()
res_bal = pd.DataFrame()
res_pos = pd.DataFrame()"""

def generate_table(df):
    return html.Div([
            dash_table.DataTable(data=df.to_dict('records'), page_size=11, style_table={'overflowX': 'auto'})
        ]),

layout = [html.Div([
    html.Br(),
    html.Div(className='twelve columns', children='BD UNAM Completo',
        style={'textAlign': 'center', 'color': 'blue', 'fontSize': 30}),
    html.Br(),
    html.Div([
        dcc.Interval(id="progress-interval", n_intervals=0, interval=400, disabled=False),
        dbc.Progress(id="progress", value = 0, label = "", animated = True)
    ]),
    html.Div([
        html.P('Cargue el archivo csv que contiene los datos'),
        dcc.Upload(id='upload-data', 
    		children=html.Div([
            	'Drag and Drop or ',
            	html.A('Select Files')
            ]), className = "twelve columns", style={
            	'height': '60px',
            	'lineHeight': '60px',
            	'borderWidth': '1px',
            	'borderStyle': 'dashed',
            	'borderRadius': '5px',
            	'textAlign': 'center',
            	'margin': '10px'},
    	),
    ]),
    html.Br(),
    html.Div(id = 'html-table', className = "twelve columns"),
    html.Br(),
    html.Div([
        html.P('Observe los dos grafos completos generados:')], className = 'twelve columns'),
    dbc.Row([
        dbc.Col([
            html.P('Grafo completo', style = {
                'textAlign': 'center',
                'fontSize': 20,
                'color': '#349000'})
        ], className = 'six columns'),
        dbc.Col([
            html.P('Sub grafo', style = {
                'textAlign': 'center',
                'fontSize': 20,
                'color': '#349000'})
        ], className = 'six columns')
    ], className = 'twelve columns'),
    html.Br(),
    html.Div([
        html.P('Aplicando los algoritmos de Machine Learning se obtiene')], className = 'twelve columns'),
    dbc.Row([
        dbc.Col([
            html.P('Scores balanceados', style = {
                'textAlign': 'center',
                'fontSize': 20,
                'color': '#349000'}),
            html.Div(id = 'res_bal'),
        ], className = 'six columns'),
        dbc.Col([
            html.P('Scores positivos', style = {
                'textAlign': 'center',
                'fontSize': 20,
                'color': '#349000'}),
            html.Div(id = 'res_pos'),
        ], className = 'six columns')
    ], className = 'twelve columns'),
    dcc.Store(id='output-data-upload')
])
]

#____________ Callbacks __________
@app.callback(Output('output-data-upload', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              prevent_initial_call = True)
def upload_datos(list_of_contents, filename):
    if list_of_contents is not None:
        try:
            content_type, content_string = list_of_contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df.to_json(date_format='iso', orient='split')
        except Exception as e:
            return html.Div([
                'There was an error processing this file.'
            ])

@app.callback(Output('html-table', 'children'),
              Input('output-data-upload', 'data'),
              prevent_initial_call = True)
def generateTableHtml(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    return generate_table(df)

@app.callback(Output('res_bal', 'children'),
              Output('res_pos', 'children'),
              Input('output-data-upload', 'data'),
              prevent_initial_call = True)
def getParametros(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    df_sample_train, df_sample_test = hs.filterAndSplit(df)
    res_bal, res_pos = models.transformVariables(df_sample_train, df_sample_test)
    return generate_table(res_bal), generate_table(res_pos)

@app.callback(
    [Output("progress", "value"), Output("progress", "label"), Output("progress-interval", "disabled"), Output("progress", "animated")],
    Input("progress-interval", "n_intervals"),
    State('upload-data', 'contents'),
    State('output-data-upload', 'data'),
    State('res_bal', 'children'))
def update_progress(n, contents, data, children):
    if contents is None:
        return 0, "", False, False
    if data is None:
        return 20, "Procesando archivo", False, True
    if children is None:
        return 40, "Grafos terminados - Compilando modelos", False, True
    else:
        return 100, "Todo listo", True, False

if __name__ == '__main__':
    app.run(debug=True)