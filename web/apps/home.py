from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from apps import home, intento3, consultas


#Contenido de tab1: carga de datos
tab1_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            html.P('''Lo primero es cargar a la página web un archivo con extensión .csv donde se encuentren todos
                los datos necesarios para crear los modelos. Cada registro del archivo representa un artículo escrito
                por un autor.'''),
            html.P('''Los campos necesarios en el archivo son: ''')]),
        dbc.Row([
            dbc.Col([
                dbc.Table([
                    #Encabezado de tabla
                    html.Thead(html.Tr([html.Th("#"),html.Th("Campo"), html.Th("Descripción")])),
                    #Cuerpo de tabla
                    html.Tbody([
                        html.Tr([html.Td("1"), html.Td("indice"), html.Td("Número que sirve de identificador")]),
                        html.Tr([html.Td("2"), html.Td("Titulo"), html.Td("Nombre del artículo científico")]),
                        html.Tr([html.Td("3"), html.Td("Año"), html.Td("Año de publicación del artículo")]),
                        html.Tr([html.Td("4"), html.Td("Autor_norm"), html.Td("Nombre del autor")]),
                        html.Tr([html.Td("5"), html.Td("Afiliacion1"), html.Td("Dependencia del autor")]),
                        html.Tr([html.Td("6"), html.Td("Afiliacion2"), html.Td("Segunda dependencia del autor")]),
                        html.Tr([html.Td("7"), html.Td("ISBN"), html.Td("ISBN del artículo")]),
                        ])
                    ], striped=True)], width=5),
            dbc.Col([
                dbc.Table([
                    #Encabezado de tabla
                    html.Thead(html.Tr([html.Th("#"),html.Th("Campo"), html.Th("Descripción")])),
                    #Cuerpo de tabla
                    html.Tbody([
                        html.Tr([html.Td("8"), html.Td("ISSN"), html.Td("ISSN del artículo")]),
                        html.Tr([html.Td("9"), html.Td("Doi"), html.Td("Doi del artículo")]),
                        html.Tr([html.Td("10"), html.Td("URL"), html.Td("Enlace al artículo en línea")]),
                        html.Tr([html.Td("11"), html.Td("Area"), html.Td("Área de la computación a la que pertenece el artículo")]),
                        html.Tr([html.Td("12"), html.Td("Subarea"), html.Td("Subareas específicas a las que pertenece el artículo")]),
                        html.Tr([html.Td("13"), html.Td("Keywords"), html.Td("Palabras clave del artículo")]),
                        html.Tr([html.Td("14"), html.Td("Abstract"), html.Td("Abstract del artículo")]),
                        ])
                    ], striped=True)], width=7),
                ]),
            html.Br(),
            dbc.Row([
                html.P('Ejemplo de una base de datos correcta:'),
                html.Img(src = '/assets/img/csv.png', width = 800)]),
            html.Br(),
            html.Br(),
            html.Div(dbc.Button("Empezar con la carga de datos", color="primary", id='button-carga', outline=True, href='/apps/intento3'), 
                className="d-grid gap-2 col-6 mx-auto"),
            html.Div(id='redirect-datos')
        ]))

#Contenido de tab2: generación de grafo
tab2_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.P(['''Al cargar el archivo, se construirá un grafo que contiene nodos Autor y Artículo, de forma
                    que cada autor se conecta con aquellos artículos que ha escrito. Los nodos ''',
                    html.Span('Autor', style = {'color':'blue'}), ' estarán coloreados en ',
                    html.Span('azul', style = {'color':'blue'}), ', mientras que los nodos ',
                    html.Span('Artículo', style = {'color':'red'}), ' estarán coloreados en ',
                    html.Span('rojo', style = {'color':'red'}), '.']),
                html.P('''El grafo es interactivo de manera que es posible acercar y alejar el grafo, hacer clic y 
                    arrastrar los nodos a distintos lugares. Cada vez que se hace clic en un nodo, se muestra la información
                    o atributos que guarda ese nodo. '''),
                html.P('Una muestra ejemplo del grafo se puede observar en la imagen:')], width = 6),
            dbc.Col([
                html.Img(src='/assets/img/grafo.png', width = '100%')
                ], width = 6)
            ])
        ]))

#Contenido de tab3: Creación de modelos
tab3_content = dbc.Card(
    dbc.CardBody([
        html.P('''En esta parte, se extraen un conjunto características o de información relevante que 
            arroja el grafo para entrenar el modelo. Para lograr la predicción de enlaces, se buscaron 
            características de proximidad, características de agregación y características topológicas.'''),
        dbc.Accordion([
            dbc.AccordionItem('''Buscan representar cierta cercanía entre dos autores. Por ejemplo, se 
                puede decir que dos autores son cercanos si han escrito artículos con varias palabras clave 
                en común. ''', title="Características de proximidad"),
            dbc.AccordionItem('''Están basadas en la suma de ciertos atributos como la suma de vecinos o de 
                artículos de un par de autores. Estas características buscan expresar una probabilidad donde 
                se piensa que mientras mayor sea esa suma, mayor es la probabilidad de que los autores estén 
                mejor conectados.''', title="Características de agregación"),
            dbc.AccordionItem('''Consisten en índices obtenidos a través de fórmulas matemáticas que buscan 
                expresar conectividad tan sólo por la forma en la que está construido el grafo.''', 
                title="Características topológicas"),
            ]),
        html.Br(),
        html.Br(),
        html.P('''Para esta aplicación en particular, por cada par de autores se obtuvieron las siguientes 
            características:'''),
        dbc.Row([
            dbc.Col(width = 2),
            dbc.Col(
                dbc.Table([
                    html.Tbody([
                        html.Tr([html.Td("Suma de vecinos"), html.Td("Suma de vecinos secundarios")]),
                        html.Tr([html.Td("Suma de palabras clave"), html.Td("Número de palabras clave en común")]),
                        html.Tr([html.Td("Índice de Clustering"), html.Td("Coeficiente de Jaccard")]),
                        html.Tr(html.Td("Índice de Asignación de Recursos tomando en cuenta la comunidad entre dos autores", colSpan=2)),
                    ])], striped=True, bordered=True), width = 8),
            ]),
        html.Br(),
        html.P('''Por último se aplicaron los siguientes algoritmos de clasificación de Aprendizaje Supervisado:'''),
        dbc.Row([
            dbc.Col(width = 2),
            dbc.Col(
                dbc.Table([
                    html.Tbody([
                        html.Tr([html.Td("Árboles de Decisión"), html.Td("Clasificador Naive-Bayes")]),
                        html.Tr([html.Td("Máquinas de Soporte Vectorial"), html.Td("Perceptrón Multicapa")]),
                        html.Tr([html.Td("K-Vecinos más cercanos"), html.Td("Red de Base Radial (RBF)")]),
                        html.Tr(html.Td("Bagging", colSpan=2)),
                    ])], striped=True, bordered=True), width = 8),
            ]),
        ]))

#Contenido de tab4: Validación de modelos
tab4_content = dbc.Card(
    dbc.CardBody([
        html.P('''En esta parte, se utilizan los modelos ya creados para probar la clasificación en registros cuya
            clasificación ya se conoce. De esa forma se puede comparar la clasificación real con aquella hecha por el
            modelo y saber si su funcionamiento es correcto. Para esto, se calculan diferentes métricas que representan
            en procentaje el correcto desempeño del modelo.'''),
        html.P('''Las métricas que se toman en cuenta en esta aplicación son:'''),
        dbc.Row([
            dbc.Col(width = 3),
            dbc.Col(
                dbc.Table([
                    html.Tbody([
                        html.Tr([html.Td("Exactitud"), html.Td("Puntuación F1")]),
                        html.Tr([html.Td("Precisión"), html.Td("Error Cuadrático Medio")]),
                        html.Tr(html.Td("Sensibilidad (Recall)", colSpan=2)),
                    ])], striped=True, bordered=True), width = 6),
            ]),
        html.P('''Se obtienen dos tipos de métricas para los modelos:'''),
        dbc.Row([
            dbc.Col(width = 1),
            dbc.Col([
                html.H6('Resultados balanceados', style = {'font-weight': 'bold'}),
                html.P("""Los resultados balanceados corresponden a las métricas de los modelos para todas las etiquetas. Es decir, las métricas
                            de desempeño aplican tanto para la predicción de autores conectados, así como para la predicción de autores no conectados
                            Cada modelo cuenta con métricas aplicables a cada etiqueta (conectados o no conectados), en donde se evaluan las instancias
                            verdaderas para cada una."""),
                html.P("""Al ser resultados balanceados, las métricas de cada etiqueta se promedian y como resultado se obtiene una medición general."""),
                ], width = 5),
            dbc.Col([
                html.H6('Resultados positivos', style = {'font-weight': 'bold'}),
                html.P("""Los resultados positivos consisten en únicamente las métricas obtenidas para cada modelo, en donde estos clasificaron con
                        éxito a los autores que realmente están conectados, ignorando por completo a las clasificaciones correctas de los autores
                        que no lo están"""),], width = 5),
            ]),
        ]))
  
#Contenido de tab5: Predicción de enlaces      
tab5_content = dbc.Card(
    dbc.CardBody([
        html.P('''Una vez que el modelo está entrenado y validado, es posible utilzarlo para generar predicciones
            acerca de qué autores es posible que colaboren en un futuro. Lo que el modelo hace es obtener las 
            características antes mencionadas para un par específico de autores que aún no hayan colaborado. Después 
            normaliza esos valores tomando en cuenta el conjunto de datos con los que se validó el modelo. Finalmente, 
            hace la predicción.'''),
        html.P(['''Para esta parte solo es necesario seleccionar dos autores haciendo uso del grafo. Una vez seleccionados,
            se hace clic en el botón ''',
            html.Span('Predecir', style = {'font-weight': 'bold'}), ' y se mostrará el resultado: ',
            html.Span('Conectados', style = {'font-weight': 'bold'}), ' o ',
            html.Span('No Conectados', style = {'font-weight': 'bold'}), '''. También se podrá desplegar una tabla con la 
            clasificación hecha por cada modelo entrenado.'''],),
        dbc.Row([
            dbc.Col(html.Img(src = '/assets/img/pred_pos.png', width = '100%'), width = 6),
            dbc.Col(html.Img(src = '/assets/img/pred_neg.png', width = '100%'), width = 6),
            ]),
        html.P('''El resultado mostrado se deduce si 4 o más modelos de los 7 utilizados coinciden en una misma clasificación. 
            Sin embargo, se recomienda revisar las métricas de validación de cada modelo para confirmar 
            su precisión y confiabilidad.'''),
        ]))

#Contenido de tab6: Consultas
tab6_content = dbc.Card(
    dbc.CardBody([
        html.P('''Aquí puedes mover e interactuar con cada componente del grafo creado. Al dar clic en un nodo o
            arista, se muestra la información que representa ese nodo o arista. También, se colorea cada nodo según
            su tipo: Artículo, Autor, Dependencia o Keyword.'''),
        html.P('''Además de eso se pueden hacer consultas a la base de datos por Autor, por Título, por Keyword y más.'''),
        html.Div(dbc.Button("Empezar con las consultas", color="primary", id='button-consultas', outline=True, href='/apps/consultas'), 
                className="d-grid gap-2 col-6 mx-auto"),
            html.Div(id='redirect-consultas')
        ]))


layout = [html.Div([
        html.Br(),
        html.H1('Ciencia de Datos en Bases de Datos Multimodelo'),
        html.H5('''Predicción de enlaces en redes sociales utilizando
                Teoría de Grafos y Aprendizaje Supervisado ''')], style={'textAlign':'center'}),

    #html.Div([
    #    html.Br(),
    #    html.P('''Intro'''),
    #    ], style = {
    #        'textAlign': 'justify',
    #        'padding-top':'20px', 
    #        'padding-left':'40px', 
    #        'padding-right':'40px', 
    #        'padding-bottom':'40px'}),

    html.Div([
        html.Br(),
        html.H4('¿De qué trata la predicción de enlaces?'),
        dbc.Row([
            dbc.Col(html.P('''Dentro del análisis de redes sociales, la tarea de predicción de enlaces representa uno de los mayores
            retos. Las redes sociales son una representación de las interacciones entre personas. Gráficamente es posible
            modelar estas interacciones a través de grafos donde cada persona es representada por un nodo y las aristas
            o lineas entre dos nodos representan alguna clase de relación entre ellos. Sin embargo, las redes sociales son 
            dinámicas en el sentido de que evolucionan a lo largo del tiempo, añadiendo y eliminando nodos y aristas. Esto
            provoca que sean complejas de entender y de modelar. El problema de predicción de enlaces tiene como objetivo 
            conocer qué probabilidad hay de que dos nodos que no están conectados en el estado actual del grafo, se relacionen 
            o conecten en un futuro.'''),width = 8),
            dbc.Col(html.Img(src='https://upload.wikimedia.org/wikipedia/commons/d/de/Social_graph.gif', width=320), width=4)
            ]),
        html.Br(),
        html.H4('¿Dónde entra el Aprendizaje Supervisado?'),
        html.P('''A través de la información que arroja tanto la base de datos como el grafo, es posible etiquetar diversos 
            datos y generar muestras de entrenamiento y prueba. Mientras la muestra de entrenamiento sirve para entrenar los 
            modelos ajustando parámetros y buscando patrones, la muestra de prueba sirve para aplicar ese modelo a nuevos datos
            cuya salida ya se conoce y verificar si los modelos hicieron la predicción correcta. Con esta validación es posible
            obtener métricas del rendimiento de los modelos como su precisión, exactitud, recall y más. Por último, una vez que 
            se tiene un modelo óptimo, se hace la predicción con nuevos datos. '''),
        ],style = {
            'textAlign': 'justify',
            'padding-top':'20px', 
            'padding-left':'40px', 
            'padding-right':'40px', 
            'padding-bottom':'40px'}),

    html.Div([
        html.H2('Etapas del Proyecto'),
        dbc.Card([
            dbc.CardHeader(
                dbc.Tabs([
                    dbc.Tab(label="Carga de datos", tab_id="tab-1"),
                    dbc.Tab(label="Generación de grafo", tab_id="tab-2"),
                    dbc.Tab(label="Creación de modelos", tab_id="tab-3"),
                    dbc.Tab(label="Validación de modelos", tab_id="tab-4"),
                    dbc.Tab(label="Predicción de enlaces", tab_id="tab-5"),
                    dbc.Tab(label="Consultas", tab_id="tab-6"),
                ], id="card-tabs", active_tab="tab-1",)),
            dbc.CardBody(html.P(id="card-content", className="card-text")),
            ])
        ],style = {
            'textAlign': 'justify',
            'padding-top':'20px', 
            'padding-left':'40px', 
            'padding-right':'40px', 
            'padding-bottom':'40px'})
    ]

@app.callback(Output("card-content", "children"), 
             [Input("card-tabs", "active_tab")])
def tab_content(active_tab):
    if active_tab == 'tab-1':
        return tab1_content
    if active_tab == 'tab-2':
        return tab2_content
    if active_tab == 'tab-3':
        return tab3_content
    if active_tab == 'tab-4':
        return tab4_content
    if active_tab == 'tab-5':
        return tab5_content
    return tab6_content

@app.callback(Output("redirect-datos", "children"), 
             Input("button-carga", "n_clicks"))
def click_carga(b_carga):
    if b_carga != None and b_carga > 0:
        return intento3.layout

@app.callback(Output("redirect-consultas", "children"), 
             Input("button-consultas", "n_clicks"))
def click_carga(b_carga):
    if b_carga != None and b_carga > 0:
        return consultas.layout
