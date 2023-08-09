from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import home, intento3

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Holis", href="/apps/intento3")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("lol", header=True),
                dbc.DropdownMenuItem("jsjsjs", href=""),
                dbc.DropdownMenuItem("aiuda", href=""),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
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
    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/intento3':
        return intento3.layout
    return home.layout


if __name__ == '__main__':
    #app.run_server(debug=False, host='0.0.0.0', port=8000)
    #app.run_server(debug=True, host='0.0.0.0', port=8000)
    app.run(debug=True)

