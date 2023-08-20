from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import pandas as pd
import networkx as nx
import numpy as np
import base64, io
import json
from apps import consultas_functions as cf

from app import app

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

def getCatalogues(G):
    list_autores = []
    list_articulos = []
    list_afiliaciones = []
    list_keywords = []
    for nodo in G.nodes():
        if G.nodes[nodo]['tipo'] == 'autor':
            list_autores.append(nodo)
        if G.nodes[nodo]['tipo'] == 'articulo':
            list_articulos.append(nodo)
        if G.nodes[nodo]['tipo'] == 'afiliacion':
            list_afiliaciones.append(nodo)
    return list_autores, list_articulos, list_afiliaciones

layout = [html.Div([
    html.H2('Consultas'),
    html.Br(),
    ], style = {
            'textAlign': 'justify',
            'padding-top':'20px', 
            'padding-left':'40px', 
            'padding-right':'40px', 
            'padding-bottom':'40px'}),
    
    html.Div([
        html.H4('Buscar por Autor, Artículo, Keyword o Afiliación'),
        html.Br(),
        html.P('''En esta sección es posible buscar información sobre autores de un artículo, autores de una afiliación, 
            artículos que contienen una keyword, artículos de un autor, keywords de un artículo y afiliaciones de un 
            autor.'''),
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.InputGroupText("Buscar"),
                    dbc.Select(options=[
                        {"label": "Autor", "value": 'autor'},
                        {"label": "Artículo", "value": 'articulo'},
                        {"label": "Afiliación", "value": 'afiliacion'},
                        {"label": "Keyword", "value": 'keyword'},
                        ], id = 'dropdown-buscar-tipo'),
                    ]),
                ],width = 3),
            dbc.Col([
                dbc.InputGroup([
                    dbc.InputGroupText("Por"),
                    html.Div(id='children-buscar-por'),
                    ]),
                ],width = 2),
            dbc.Col([
                html.Div(id='children-buscar'),
                ], width = 6),
            dbc.Col([
                dbc.Button("Buscar", id="button-buscar", n_clicks=0, disabled=True),
                ], width = 1)

            ]),
        html.Br(),
        html.Div(id='resultados-busqueda'),
        ], style = {
            'textAlign': 'justify',
            'padding-top':'20px', 
            'padding-left':'40px', 
            'padding-right':'40px', 
            'padding-bottom':'40px'}),
    
    html.Div([
        html.H4('Consultas por Frecuencia'),
        html.Br(),
        html.P('''Para estas consultas es posible conocer cuales son los autores, artículos, afiliaciones, etc. que
            tienen la mayor cantidad de elementos. Por ejemplo: ¿qué artículo tiene más autores?'''),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("Preguntas: "),
                    dbc.Select(options=[
                        {"label": "¿Cuáles son las keywords más utilizadas?", "value": 1},
                        {"label": "¿Qué área de conocimiento tiene más artículos?", "value": 2},
                        {"label": "¿Qué autor tiene más colaboradores?", "value": 3},
                        {"label": "¿Qué afiliación tiene a los autores más productivos?", "value": 4},
                        {"label": "¿Qué artículo tiene más autores?", "value": 5},
                        ], id = 'dropdown-consultas-frec'),
                    ]), width = 11),
            dbc.Col(dbc.Button("Buscar", id="button-consultas-frec", n_clicks=0, disabled=True), width = 1),
            ]),
        html.Br(),
        html.Div(id='resultados-consultas-frec'),
        ], style = {
            'textAlign': 'justify',
            'padding-top':'20px', 
            'padding-left':'40px', 
            'padding-right':'40px', 
            'padding-bottom':'40px'}),

    html.Div([
        html.H4('Similitud entre Artículos'),
        html.Br(),
        html.P('''Dado dos artículos, se muestra la similitud entre ambos. La similitud se representa como un número 
            de 0 al 1, donde 1 siginifica que los artículos son iguales y 0 que no existe ninguna similitud.'''),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("Artículo 1 "),
                    dbc.Select(options = [], id = 'dropdown-similitud-art1'),
                    ]), width = 5),
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("Artículo 2 "),
                    dbc.Select(options = [], id = 'dropdown-similitud-art2'),
                    ]), width = 5),
            dbc.Col(
                dbc.Button("Buscar", id="button-similitud", n_clicks=0, disabled=True, style = {'textAlign': 'justify'}), 
                width = 2),
            ]),
        html.Br(),
        html.Div(id='resultados-similitud'),
        ], style = {
            'textAlign': 'justify',
            'padding-top':'20px', 
            'padding-left':'40px', 
            'padding-right':'40px', 
            'padding-bottom':'40px'}),
    ]


#____________ Callbacks __________
#____________ Opciones segundo dropdown: Buscar por __________
@app.callback(Output('children-buscar-por', 'children'),
              Input('dropdown-buscar-tipo', 'value'))
def update_opciones_buscar_por(nodeType):
    if nodeType is None:
        options = []
    if nodeType == 'autor':
        options = [{"label": "Afiliación", "value": 'afiliacion'},{"label": "Artículo", "value": 'articulo'}]
    if nodeType == 'articulo':
        options = [{"label": "Keyword", "value": 'keyword'},{"label": "Autor", "value": 'autor'}]
    if nodeType == 'afiliacion':
        options = [{"label": "Autor", "value": 'autor'}]
    if nodeType == 'keyword':
        options = [{"label": "Artículo", "value": 'articulo'}]
    return dbc.Select(options=options, id = 'dropdown-buscar-por')

#____________ Opciones tercer dropdown: Buscar __________
@app.callback(Output('children-buscar', 'children'),
              Input('dropdown-buscar-por', 'value'),
              Input('graphCompleto', 'data'))
def update_opciones_buscar(nodeType, cy_G):
    if cy_G is not None:
        G = convertCYToGraph(cy_G)
        list_autores, list_articulos, list_afiliaciones = getCatalogues(G)
        if nodeType is None:
            options = [] 
        if nodeType == 'autor':
            options = [{'label':x, 'value':x} for x in list_autores]
        if nodeType == 'articulo':
            options = [{'label':x, 'value':x} for x in list_articulos]
        if nodeType == 'afiliacion':
            options = [{'label':x, 'value':x} for x in list_afiliaciones]
        if nodeType == 'keyword':
            return dbc.Input(type='text', id='dropdown-buscar')
        return dbc.Select(options=options, id='dropdown-buscar')

#____________ Habilitar botón Buscar __________
@app.callback(Output('button-buscar','disabled'),
              Input('dropdown-buscar', 'value'))
def activate_button_buscar(value):
    if value == None:
        return True

#____________ Mostrar resultados de búsqueda __________
@app.callback(Output('resultados-busqueda', 'children'),
              Input('graphCompleto', 'data'),
              Input('button-buscar', 'n_clicks'),
              State('dropdown-buscar-tipo', 'value'),
              State('dropdown-buscar-por', 'value'),
              State('dropdown-buscar', 'value'))
def buscar(cy_G, nBuscar, nodeType, buscarPor, atribute):
    if nodeType is None or buscarPor is None or atribute is None or cy_G is None:
        return dbc.Alert('Faltan datos para hacer la consulta', color = 'danger')
    else:
        G = convertCYToGraph(cy_G)
        list_nodes = cf.getNodesQuery(G, nodeType, atribute)
        
        if nodeType == 'autor':
            html_layout = [html.P(['Los ', html.Span('autores', style = {'font-weight': 'bold'}), ' encontrados para ',
                html.Span(atribute, style = {'font-weight': 'bold'}), ' son: '])]
            for element in list_nodes:
                html_layout.append(html.Li(element))

        if nodeType == 'articulo':
            html_layout = [html.P(['Los ', html.Span('artículos', style = {'font-weight': 'bold'}), ' encontrados para ',
                html.Span(atribute, style = {'font-weight': 'bold'}), ' son: '])]
            for element in list_nodes:
                info = cf.getInfoArticle(G, element)
                html_layout.append(html.Li(element))
                #for item in info.items():
                #    if item[0] == 'keywords':
                #        html_layout.append(html.P(item[0]))
                #        for x in item[1]:
                #            html_layout.append(html.P(x))
                #    else:
                #        html_layout.append(html.P(item[0] + ' ' + item[1]))

        if nodeType == 'afiliacion':
            html_layout = [html.P(['Las ', html.Span('afiliaciones', style = {'font-weight': 'bold'}), ' encontradas para ',
                html.Span(atribute, style = {'font-weight': 'bold'}), ' son: '])]
            for element in list_nodes:
                html_layout.append(html.Li(element))

        if nodeType == 'keyword':
            html_layout = [html.P(['Las ', html.Span('keywords', style = {'font-weight': 'bold'}), ' encontradas para ',
                html.Span(atribute, style = {'font-weight': 'bold'}), ' son: '])]
            for element in list_nodes:
                html_layout.append(html.Li(element))

        return dbc.Card(dbc.CardBody(html_layout))

#____________ Habilitar botón Buscar __________
@app.callback(Output('button-consultas-frec','disabled'),
              Input('dropdown-consultas-frec', 'value'))
def activate_button_buscar(value):
    if value == None:
        return True

#____________ Mostrar resultados de consulta de frecuencia __________
@app.callback(Output('resultados-consultas-frec', 'children'),
              Input('graphCompleto', 'data'),
              Input('graphSubCompleto', 'data'),
              Input('button-consultas-frec', 'n_clicks'),
              State('dropdown-consultas-frec','value'))
def consultas_frec(cy_G, cy_G_sub, nConsultas, value):
    if cy_G is not None and cy_G_sub is not None and value is not None:
        G = convertCYToGraph(cy_G)
        G_sub = convertCYToGraph(cy_G_sub)
        html_layout = []
        i = 0

        if value == '1':
            res = cf.keywordFrecuency(G)
        if value == '2':
            res = cf.areaFrecuency(G)
        if value == '3':
            res = cf.coAuthors(G_sub)
        if value == '4':
            res = cf.dependenciaFrecuency(G)
        if value == '5':
            res = cf.articuloAutorFrecuency(G)

        for item in res.items():
            if i == 5:
                break
            html_layout.append(html.Li(item[0] +': ' +str(item[1])))
            i += 1
        return dbc.Card(dbc.CardBody(html_layout))   

#____________ Opciones dropdown para Similitud __________
@app.callback(Output('dropdown-similitud-art1','options'),
              Output('dropdown-similitud-art2','options'),
              Input('graphCompleto', 'data'))
def opciones_similitud(cy_G):
    if cy_G is not None:
        G = convertCYToGraph(cy_G)
        list_autores, list_articulos, list_afiliaciones = getCatalogues(G)
        options = [{'label':x, 'value':x} for x in list_articulos] 
        return options, options 

#____________ Habilitar botón Similitud __________
@app.callback(Output('button-similitud','disabled'),
              Input('dropdown-similitud-art1', 'value'),
              Input('dropdown-similitud-art2', 'value'))
def activate_button_buscar(value1, value2):
    if value1 == None or value2 == None:
        return True

