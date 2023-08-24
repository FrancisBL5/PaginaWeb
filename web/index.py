from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import home, intento3, consultas

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Grafo y Predicción", href="/apps/intento3")),
        dbc.NavItem(dbc.NavLink("Consultas", href="/apps/consultas")),
        #dbc.DropdownMenu(
        #    children=[
        #        dbc.DropdownMenuItem("lol", header=True),
        #        dbc.DropdownMenuItem("jsjsjs", href=""),
        #        dbc.DropdownMenuItem("aiuda", href=""),
        #    ],
        #    nav=True,
        #    in_navbar=True,
        #    label="More",
        #),
    ],
    brand="BD UNAM",
    brand_href="/apps/home",
    color="primary",
    dark=True,
)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Br(),
    html.Div(id='page-content', children=[]),
    # Almacenamiento de datos
    dcc.Store(id='output-data-upload'),  # DataFrame original
    dcc.Store(id='sample_train'),        # DataFrame de train
    dcc.Store(id='sample_test'),         # DataFrame de test
    dcc.Store(id='nodes_catalogue'),     # DataFrame de los node catalogue
    dcc.Store(id='graphCompleto'),       # Grafo completo
    dcc.Store(id='graphSubCompleto'),    # Sub grafo completo
    dcc.Store(id='graphTrain'),          # Grafo de train
    dcc.Store(id='graphTrainSub'),       # Grafo de train - sub
    dcc.Store(id='graphTest'),           # Grafo de test
    dcc.Store(id='graphTestSub'),         # Grafp de test - sub
    dcc.Store(id='matrizTFIDF-global'),
    dcc.Store(id='BiGrams-list')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/intento3':
        return intento3.layout
    if pathname == '/apps/consultas':
        return consultas.layout
    return home.layout


if __name__ == '__main__':
    #app.run_server(debug=False, host='0.0.0.0', port=8000)
    #app.run_server(debug=True, host='0.0.0.0', port=8000)
    app.run(debug=True)

