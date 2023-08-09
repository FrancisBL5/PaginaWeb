from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from apps import home, intento3

layout = [html.Div([
        html.Br(),
        html.H1('Ciencia de Datos en Bases de Datos Multimodelo'),
        html.H5('''Predicción de enlaces en redes sociales utilizando
                Teoría de Grafos y Aprendizaje Supervisado ''')], style={'textAlign':'center'}),

    html.Div([
        html.Br(),
        html.P('''Intro'''),
        ], style = {
            'textAlign': 'justify',
            'padding-top':'20px', 
            'padding-left':'40px', 
            'padding-right':'40px', 
            'padding-bottom':'40px'}),

    html.Div([
        html.Br(),
        html.H4('¿De qué trata la predicción de enlaces?'),
        html.P('''Dentro del análisis de redes sociales, la tarea de predicción de enlaces representa uno de los mayores
            retos. Las redes sociales son una representación de las interacciones entre personas. Gráficamente es posible
            modelar estas interacciones a través de grafos donde cada persona es representada por un nodo y las aristas
            o lineas entre dos nodos representan alguna clase de relación entre ellos. Sin embargo, las redes sociales son 
            dinámicas en el sentido de que evolucionan a lo largo del tiempo, añadiendo y eliminando nodos y aristas. Esto
            provoca que sean complejas de entender y de modelar. El problema de predicción de enlaces tiene como objetivo 
            conocer qué probabilidad hay de que dos nodos que no están conectados en el estado actual del grafo, se relacionen 
            o conecten en un futuro.'''),
        html.Br(),
        html.H4('¿Dónde entra el Aprendizaje Supervisado?'),
        html.P('''A través de la información que arroja tanto la base de datos como el grafo, es posible etiquetar diversos 
            datos y generar muestras de entrenamiento y prueba. Mientras la muestra de entrenamiento sirve para entrenar los 
            modelos ajustando parámetros y buscando patrones, la muestra de prueba sirve para aplicar ese modelo a nuevos datos
            cuya salida ya se conoce y verificar si los modelos hicieron la predicción correcta. Con esta validación es posible
            obtener métricas del rendimiento de los modelos como su precisión, exactitud, recall y más. Por último, una vez que 
            se tiene un modelo óptimo, se hace la predicción con nuevos datos. '''),
        html.Br(),
        html.H4('Pasos'),
        html.Div(
            dbc.Tabs([
                dbc.Tab([
                    dbc.Row([
                        html.P('''Lo primero es cargar a la página web un archivo con extensión .csv donde se encuentren todos
                            los datos necesarios para crear los modelos. Cada registro del archivo representa un artículo escrito
                            por un autor.'''),
                        html.P('''Los campos necesarios en el archivo son: ''')]),
                    dbc.Row([
                        dbc.Col([
                            html.Ol([
                                html.Li('Índice: numero entero que sirve de identificador'),
                                html.Li('Título: nombre del artículo científico'),
                                html.Li('Año: año de publicación del artículo'),
                                html.Li('Autor_norm: nombre del autor'),
                                html.Li('Afiliacion1: dependencia del autor'),
                                html.Li('Afiliación2: segunda dependencia del autor'),
                                html.Li('ISBN: ISBN del artículo')])]),
                        dbc.Col([
                            html.Ol([
                                html.Li('ISSN: ISSN del artículo'),
                                html.Li('Doi: Doi del artículo'),
                                html.Li('URL: enlace al artículo en línea'),
                                html.Li('Area: Área de la computación a la que pertenece el artículo'),
                                html.Li('Subarea: Subareas específicas a las que pertenece el artículo'),
                                html.Li('Keywords: palabras clave del artículo'),
                                html.Li('Abstract: abstract del artículo')], start = '7')])
                            ]),
                    dbc.Row(html.Img(src = '/assets/img/csv.png', height = 300)),
                    ], label="Carga de Datos"),
                dbc.Tab([
                    html.P('''Al cargar el archivo, se construirán dos grafos. El primero contendrá nodos de
                        Autores, Artículos, Keywords y Afiliaciones de forma que un Autor se conecta con sus 
                        Afiliaciones y con los Artículos que ha escrito; mientras que los Artículos se conectan 
                        con su Autor y sus Keywords.'''),
                    html.P('''El segundo grafo es un subgrafo del primero, donde solo existen nodos Autor. Las 
                        aristas que unen a los nodos representan aquellos Artículos que ambos autores escribieron.
                        ''')], label="Generación de grafos"),
                dbc.Tab([
                    html.P('''En esta parte, se extraen un conjunto características o de información relevante que 
                        arroja el grafo para entrenar el modelo. Para lograr la predicción de enlaces, se buscaron 
                        características de proximidad, características de agregación y características topológicas.'''),
                    html.P('''En las características de proximidad se buscó la representación de cierta cercanía entre
                        dos autores; por ejemplo, se puede decir que dos autores son cercanos si han escrito artículos 
                        con varias palabras clave en común. '''),
                    html.P('''Las características de agregación están basadas en la suma de ciertos atributos como: suma
                        de vecinos o de artículos de un par de autores. Estas características buscan expresar una 
                        probabilidad donde se piensa que mientras mayor sea esa suma, mayor es la probabilidad de que los 
                        autores estén mejor conectados.'''),
                    html.P('''Por último, las características topológicas consisten en índices obtenidos a través de fórmulas
                        matemáticas que buscan expresar conectividad tan sólo por la forma en la que está construido el grafo.'''),
                    html.P('''Para esta aplicación en particular se utilizaron las siguientes características:'''),
                    html.Li('Suma de vecinos de un par de autores'),
                    html.Li('Suma de los vecinos secundarios de un par de autores'),
                    html.Li('Suma de las palabras clave de par de autores'),
                    html.Li('Número de palabras clave en común de un par de autores'),
                    html.Li('Índice de clustering entre dos autores'),
                    html.Li('Coeficiente de Jaccard entre dos autores'),
                    html.Li('Índice de Asignación de Recursos tomando en cuenta la comunidad entre dos autores'),
                    html.Br(),
                    html.P('''Por último se aplicaron los siguientes algoritmos de clasificación de Aprendizaje Supervisado:'''),
                    html.Li('Árboles de decisión'),
                    html.Li('Máquinas de soporte vectorial'),
                    html.Li('K-Vecinos más cercanos'),
                    html.Li('faltan 3 '),
                    ], label = "Creación de modelos predictivos"),
                dbc.Tab([
                    html.P("This is the content of the third section")], label = "Validación de modelos"),
                dbc.Tab([
                    html.P("This is the content of the third section")], label = "Predicción de enlaces"),
                dbc.Tab([
                    html.P("This is the content of the third section")], label = "Consultas con grafo interactivo"),
                ]))

        ],style = {
            'textAlign': 'justify',
            'padding-top':'20px', 
            'padding-left':'40px', 
            'padding-right':'40px', 
            'padding-bottom':'40px'})
    ]
