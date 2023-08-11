from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import pandas as pd
import networkx as nx
import base64, io
import hassan as hs
import modelos as models

app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
app.title = 'Página principal'

def generate_table(df):
    """ Función que genera la tabla en html a partir de un DataFrame """
    return html.Div([
            dash_table.DataTable(data=df.to_dict('records'), page_size=11, style_table={'overflowX': 'auto'})
        ]),

def convertGraphToCY(G):
    """Función que transforma un grafo de NetworkX en un Cytoscape JSON format"""
    cy = nx.readwrite.json_graph.cytoscape_data(G)
    for i in cy['elements']['nodes']:
        nodo = i['data']['name']
        try:
            tipo = G.nodes[nodo]['tipo']
            if tipo == 'articulo':
                i.update({'style': {'background-color': '#FF0000'}})
            elif tipo == 'autor':
                i.update({'style': {'background-color': '#0000FF'}})
            elif tipo == 'keyword':
                i.update({'style': {'background-color': '#00FF00'}})
            elif tipo == 'afiliacion':
                i.update({'style': {'background-color': '#FFFF00'}})
            else:
                i.update({'style': {'background-color': '#0F0000'}})
            return cy
        except Exception as e:
            i.update({'style': {'background-color': '#0000FF'}})
    return cy

def convertCYToGraph(cy):
    """Función que transforma un Cytoscape JSON y devuelve un grafo G de NetworkX"""
    G = nx.Graph()
    for list_nodo in cy['elements']['nodes']:
        nodo = dict(list_nodo['data'])
        G.add_node(nodo['value'])
        for atributo, valor in nodo.items():
            if atributo in ['id', 'name', 'value']:
                continue
            G.nodes[nodo['value']][atributo] = valor
    for list_edges in cy['elements']['edges']:
        edge = dict(list_edges['data'])
        G.add_edge(edge['source'], edge['target'])
    return G

app.layout = html.Div([
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
    dcc.Store(id='output-data-upload'),  # DataFrame original
    dcc.Store(id='sample_train'),        # DataFrame de train
    dcc.Store(id='sample_test'),         # DataFrame de test
    dcc.Store(id='nodes_catalogue'),     # DataFrame de los node catalogue
    dcc.Store(id='graphCompleto'),       # Grafo completo
    dcc.Store(id='graphSubCompleto'),    # Sub grafo completo
    dcc.Store(id='graphTrain'),          # Grafo de train
    dcc.Store(id='graphTrainSub'),       # Grafo de train - sub
    dcc.Store(id='graphTest'),           # Grafo de test
    dcc.Store(id='graphTestSub')         # Grafp de test - sub
])

#____________ Callbacks __________
#____________ Recibir el archivo inicial
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

#____________ Generar tabla del archivo inicial
@app.callback(Output('html-table', 'children'),
              Input('output-data-upload', 'data'),
              prevent_initial_call = True)
def generateTableHtml(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    return generate_table(df)

#____________ Obtener grafos
@app.callback(Output('graphCompleto', 'data'),
              Output('graphSubCompleto', 'data'),
              Output('graphTrain', 'data'),
              Output('graphTrainSub', 'data'),
              Output('graphTest', 'data'),
              Output('graphTestSub', 'data'),
              Output('nodes_catalogue', 'data'),
              Input('output-data-upload', 'data'),
              prevent_initial_call = True)
def getAllGraphs(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    G_completo, G_sub_completo, G_train, G_sub_train, G_test, G_sub_test, nodes_catalogue = hs.filterAndSplit(df)
    cy_G_completo = convertGraphToCY(G_completo)
    cy_G_sub_completo = convertGraphToCY(G_sub_completo)
    cy_G_train = convertGraphToCY(G_train)
    cy_G_sub_train = convertGraphToCY(G_sub_train)
    cy_G_test = convertGraphToCY(G_test)
    cy_G_sub_test = convertGraphToCY(G_sub_test)
    return cy_G_completo, cy_G_sub_completo, cy_G_train, cy_G_sub_train, cy_G_test, cy_G_sub_test, nodes_catalogue.to_json(date_format='iso', orient='split')
    #df_sample_train, df_sample_test = hs.filterAndSplit(df)
    #res_bal, res_pos = models.transformVariables(df_sample_train, df_sample_test)
    #return generate_table(res_bal), generate_table(res_pos)

# ___________ Obtener samples
@app.callback(Output('sample_train', 'data'),
              Output('sample_test', 'data'),
              Input('graphTrain', 'data'),
              Input('graphTest', 'data'),
              Input('nodes_catalogue', 'data'),
              Input('graphTrainSub', 'data'),
              Input('graphTestSub', 'data'),
              prevent_initial_call = True)
def getSamples(cy_G_train, cy_G_test, nodes_catalogue_json, cy_G_sub_train, cy_G_sub_test):
    G_train = convertCYToGraph(cy_G_train)
    G_test = convertCYToGraph(cy_G_test)
    G_sub_train = convertCYToGraph(cy_G_sub_train)
    G_sub_test = convertCYToGraph(cy_G_sub_test)
    nodes_catalogue = pd.read_json(nodes_catalogue_json, orient='split')
    print("Nodes catalogue listo")
    sample_train, sample_test = hs.createSamples(G_train, G_test, nodes_catalogue)
    print("Samples vacios")
    sample_train, sample_test = hs.getParameters(sample_train, sample_test, G_train, G_test, G_sub_train, G_sub_test)
    print("Samples llenos")
    return sample_train.to_json(date_format='iso', orient='split'), sample_test.to_json(date_format='iso', orient='split')

# ___________ Obtener Medidas de los modelos
@app.callback(Output('res_bal', 'children'),
              Output('res_pos', 'children'),
              Input('sample_train', 'data'),
              Input('sample_test', 'data'),
              prevent_initial_call = True)
def getMedidas(sample_train_json, sample_test_json):
    df_sample_train = pd.read_json(sample_train_json, orient='split')
    df_sample_test = pd.read_json(sample_test_json, orient='split')
    res_bal, res_pos = models.transformVariables(df_sample_train, df_sample_test)
    return generate_table(res_bal), generate_table(res_pos)

@app.callback(
    [Output("progress", "value"), Output("progress", "label"), Output("progress-interval", "disabled"), Output("progress", "animated")],
    Input("progress-interval", "n_intervals"),
    State('upload-data', 'contents'),
    State('output-data-upload', 'data'),
    State('graphCompleto', 'data'),
    State('sample_train', 'data'),
    State('res_bal', 'children'))
def update_progress(n, dato_cargado, dato_html, grafo, sample, respuesta):
    if dato_cargado is None:
        return 0, "", False, False
    if dato_html is None:
        return 15, "Procesando archivo", False, True
    if grafo is None:
        return 30, "Generando grafos", False, True
    if sample is None:
        return 45, "Generando samples", False, True
    if respuesta is None:
        return 60, "Compilando modelos", False, True
    else:
        return 100, "Todo listo", True, False

if __name__ == '__main__':
    app.run(debug=True)