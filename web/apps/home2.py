from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from app import app
from apps import home

layout = [html.Div([
    html.Br(),
    html.Div([
    	html.Div(html.Br(), className = "four columns"),
   		html.H1('Smart Learner', style={
	    	'textAlign': 'center',
	    	'color':'white',
	    	'backgroundColor': '#303F9F',
	    	'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
	    	'border-radius':'40px',
	    	'padding':'10px',
	    	'border-width':'medium'}, className = "four columns"),], 
    	style={'textAlign':'center'}, className = "twelve columns")]),

	html.H3('¿Qué es Smart Learner?', style={
		    	'textAlign': 'center',
		    	'color':'white',
		    	'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
		    	'backgroundColor': '#26A69A',
		    	'border-radius':'30px',
		    	'padding':'5px',
		    	'border-width':'medium'}, className = "four columns"),
    html.Div([
       	html.P('''Smart Learner es una herramienta tecnológica de Inteligencia Artificial que permite 
       		la implementación de algoritmos dentro del Machine Learning.'''),
      	html.P('''Smart Learner ofrece la oportunidad de aplicar algoritmos de Asociación, Clustering
   			Clasificación y Regresión a un conjunto de datos definidos por el usuario. De esa forma, 
   			se puede obtener información acerca de los datos a través de las gráficas, tablas e imagenes
       		que aporta Smart Learner. '''),
       	html.P('''Lo más especial sobre esta herramienta es que el usuario puede modificar la mayor parte de los
       		parámetros con los que se construyen los modelos de los algoritmos para así observar cómo cambia la 
       		información desplegada así como entender de mejor manera cómo funciona cada algoritmo.''')
   		], className = "twelve columns", style = {
    		'padding-top':'20px', 
    		'padding-left':'40px', 
    		'padding-right':'40px', 
    		'padding-bottom':'40px'}),


    html.H3('¿Qué algoritmos incluye Smart Learner?', style={
		'textAlign': 'center',
    	'color':'white',
    	'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
		'backgroundColor': '#26A69A',
	   	'border-radius':'30px',
	   	'padding':'5px',
	    'border-width':'medium'}, className = "six columns"),

    html.Div([
    	html.Div([
	    	html.P('Dentro de Smart Learner se encuentran los siguientes algoritmos:', 
	    		style = {'textAlign': 'justify', 'padding':'20px'}),
	    	html.Ol([
	    		html.Div([
		    		html.Div([
		    			html.H5(html.Li('Asociación:'), style= {'textAlign': 'center'}),
		    			html.Ul(
		    		 		html.Li('Apriori'), style = {'textAlign': 'left'})
		    			], style = {
		    				'textAlign':'center',
		    				'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
		    				'padding':'20px',
		    				'backgroundColor': '#536DFE',
		    				'color':'white',
		    				'border-radius':'30px',}),
		    		html.Br(),
		    		html.Div([
			    		html.H5(html.Li('Clasificación:'), style= {'textAlign': 'center'}),
			    		html.Ul([
			    			html.Li('Regresión Logística'),
			    		 	html.Li('Árboles de Decisión'),
			    		 	html.Li('Bosques Aleatorios')
			    			])
			    		], style = {
		    				'padding':'30px',
		    				'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
		    				'backgroundColor': '#536DFE',
		    				'color':'white',
		    				'border-radius':'30px',}),
	    		], className = "four columns"),
	    		html.Div([ 
		    		html.H5(html.Li('Métricas de Distancia:'), style= {'textAlign': 'center'}),
		    		html.Ul([
		    		 	html.Li('Distancia Euclidiana'),
		    		 	html.Li('Distancia de Chebyshev'),
		    		 	html.Li('Distancia de Manhattan'),
		    		 	html.Li('Distancia de Minkowski'),
		    			]),
		    		], className = "four columns", style = {
	    				'padding':'20px',
	    				'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
	    				'backgroundColor': '#536DFE',
	    				'color':'white',
	    				'border-radius':'30px',}),
	    		html.Div([
		    		html.H5(html.Li('Clustering:'), style= {'textAlign': 'center'}),
		    		html.Ul([
		    		 	html.Li('Clustering Jerárquico'),
		    		 	html.Li('Clústering Particional'),
		    			]),
	    			], className = "four columns", style = {
	    				'padding':'20px',
	    				'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
	    				'backgroundColor': '#536DFE',
	    				'color':'white',
	    				'border-radius':'30px',}),
	    		html.Div(html.Br(), className = "two columns"),
	    		html.Br(),
	    		html.Br(),
	    		html.Br(),
	    		html.Div([
		    		html.H5(html.Li('Pronóstico:'), style= {'textAlign': 'center'}),
		    		html.Ul([
		    		 	html.Li('Árboles de Decisión'),
		    		 	html.Li('Bosques Aleatorios'),
		    			]),
		    		], className = "four columns", style = {
	    				'padding':'20px',
	    				'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
	    				'backgroundColor': '#536DFE',
	    				'color':'white',
	    				'border-radius':'30px',}),
    			], className = "twelve columns")
    		], className = "twelve columns"), 
    	], className = "twelve columns", style = {
    		'padding-left':'40px', 
    		'padding-right':'40px', 
    		'padding-bottom':'40px'}),

    html.H3('¿Cómo utilizar Smart Learner?', style={
		    	'textAlign': 'center',
		    	'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
		    	'color':'white',
		    	'backgroundColor': '#26A69A',
		    	'border-radius':'30px',
		    	'padding':'5 px',
		    	'border-width':'medium'}, className = "five columns"),

    html.Div([
    	html.P('''Para empezar se debe escoger una de las opciones mostradas en los siguientes botones o
    		al inicio de esta página.'''),
    	html.P('''Cada botón lleva a otra página que implementa el algoritmo seleccionado. Al principio de 
    		cada página se incluye de manera general cómo funciona el algoritmo y características importantes
    		de este.'''),
    	html.P('''Cada algoritmo pedirá el ingreso de un archivo con extensión .CSV para funcionar. Si no se 
    		ingresan datos adecuados el algoritmo no podrá funcionar correctamente. Es importante que para el 
    		algoritmo Apriori se ingresen datos transaccionales.'''),
    	html.P('''Se mostrarán los datos ingresados en una tabla y después para algunos algoritmos se deberá hacer 
    		una selección de variables. Para ello, se mostrará un mapa de calor que muestre la correlación entre 
    		variables. Se recomienda eliminar aquellas que tengan correlación alta (de 1 a 0.67 y -0.67 a -1)
    		con todas las variables y que no sean fundamentales para el análisis. Si no se desea eliminar ninguna 
    		variable tan sólo se deberán seleccionar todas en la lista desplegable.'''),
    	html.P('''Luego se pedirán otros datos importantes para el funcionamiento del algoritmo ya sea por medio 
    		de deslizadores, entradas por teclado numéricas, o por listas desplegables. Por medio de botones es 
    		que Smart Learner puede analizar los datos ingresados por lo que es importante presionarlos donde se 
    		encuentre uno.'''),
    	html.P('''Para finalizar, en algunos algoritmos se podrán hacer pruebas como Sistemas de Inferencia, Cálculos
    		de Distancia o se podrán descargar las gráficas.''')
    	], className = "twelve columns", style = {
    		'textAlign':'justify',
    		'padding-top':'20px',
    		'padding-left':'40px', 
    		'padding-right':'40px', 
    		'padding-bottom':'20px'}),

    html.H3('Elige un algoritmo:', className = "four columns"),

    html.Div([
        dcc.Link(html.Button(id='button-apriori', n_clicks=0, children='Algoritmo Apriori', style = {
        	'backgroundColor':'black',
        	'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
        	'color':'white',
        	'border': '2px solid white',
        	'border-radius':'20px',
        	}), href='/apps/apriori',),
    	dcc.Link(html.Button(id='button-distancia', n_clicks=0, children='Métricas de Distancia', style={
    		'backgroundColor':'black',
    		'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
        	'color':'white',
        	'border': '2px solid white',
        	'border-radius':'20px',
    		}), href='/apps/distancia'),
    	dcc.Link(html.Button(id='button-clustering', n_clicks=0, children='Clustering', style={
    		'backgroundColor':'black',
    		'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
        	'color':'white',
        	'border': '2px solid white',
        	'border-radius':'20px',
    		}), href='/apps/clusterj'),
    	dcc.Link(html.Button(id='button-regresion', n_clicks=0, children='Regresión Logística', style={
    		'backgroundColor':'black',
    		'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
        	'color':'white',
        	'border': '2px solid white',
        	'border-radius':'20px',
    		}), href='/apps/regresion'),
    	dcc.Link(html.Button(id='button-arboles', n_clicks=0, children='Árboles de Decisión', style={
    		'backgroundColor':'black',
    		'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
        	'color':'white',
        	'border': '2px solid white',
        	'border-radius':'20px',
    		}), href='/apps/arboles'),
    	dcc.Link(html.Button(id='button-bosques', n_clicks=0, children='Bosques Aleatorios', style={
    		'backgroundColor':'black',
    		'box-shadow': '0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)',
        	'color':'white',
        	'border': '2px solid white',
        	'border-radius':'20px',
    		}), href='/apps/bosques')
    	], className = "twelve columns", style = {
    		'textAlign':'justify',
    		'padding':'20px'}),

    html.Div(id='redireccion', children=[]),
    html.Hr(className = "twelve columns"),
    html.Div([
    	html.H5('Más información...'),
    	html.P('''Esta herramienta fue creada por Jessica Zepeda Baeza como proyecto de final de la materia Inteligencia 
    		Artificial del semestre 2022-2.'''),
    	], className = "twelve columns", style = {'padding-left':'20px'})
    ]

@app.callback(Output('redireccion', 'children'),
    		  Input('button-apriori', 'n_clicks'),
			  Input('button-distancia', 'n_clicks'),
			  Input('button-clustering', 'n_clicks'),
			  Input('button-regresion', 'n_clicks'),
			  Input('button-arboles', 'n_clicks'),
			  Input('button-bosques', 'n_clicks'))
def displayClick(b_apriori, b_distancia, b_clustering, b_regresion, b_arboles, b_bosques):
    if b_apriori > 0:
        return apriori.layout
    elif b_distancia > 0:
        return distancia.layout
    elif b_clustering > 0:
        return clusterj.layout
    elif b_regresion > 0:
        return regresion.layout
    elif b_arboles > 0:
        return arboles.layout
    elif b_bosques > 0:
        return bosques.layout
