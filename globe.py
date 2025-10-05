import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

app = dash.Dash(__name__)

lons = [-74.0060, 2.3522, 139.6917, 77.1025]
lats = [40.7128, 48.8566, 35.6895, 28.7041]
names = ["New York", "Paris", "Tokyo", "Delhi"]

def make_figure(rot_lon=0, rot_lat=0):
    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon=lons,
                lat=lats,
                text=names,
                mode='markers',
                marker=dict(size=7, line=dict(width=1), opacity=0.9),
                hovertemplate="<b>%{text}</b><br>Lon: %{lon:.2f}°<br>Lat: %{lat:.2f}°<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            projection=dict(type='orthographic', rotation=dict(lon=rot_lon, lat=rot_lat)),
            showland=True, landcolor="rgb(217,217,217)",
            showocean=True, oceancolor="rgb(204,230,255)",
            showcountries=True, countrycolor="rgb(180,180,180)",
            lakecolor="rgb(204,230,255)",
            showlakes=True,
            coastlinecolor="rgb(150,150,150)"
        ),
    )
    return fig

app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif"},
    children=[
        html.H2("Interactive Globe (Dash + Plotly)"),
        dcc.Graph(id="globe", figure=make_figure()),
        html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "center", "marginTop": "8px"},
            children=[
                html.Button("Start / Stop Spin", id="toggle", n_clicks=0),
                dcc.Slider(id="speed", min=0, max=20, step=1, value=5,
                           marks={0:"0", 5:"5", 10:"10", 15:"15", 20:"20"}),
                html.Div(id="info", style={"marginLeft": "12px", "minWidth": "220px"}),
            ],
        ),
        dcc.Interval(id="spin", interval=100, n_intervals=0, disabled=False),
        dcc.Store(id="rot-lon", data=0),
        dcc.Store(id="rot-lat", data=0),
    ]
)

# Spin toggle
@app.callback(
    Output("spin", "disabled"),
    Input("toggle", "n_clicks"),
    prevent_initial_call=False
)
def toggle_spin(n):
    # Odd -> disabled, Even -> enabled
    return (n or 0) % 2 == 1

# Rotation updater
@app.callback(
    Output("globe", "figure"),
    Output("rot-lon", "data"),
    Input("spin", "n_intervals"),
    Input("speed", "value"),
    State("rot-lon", "data"),
    State("rot-lat", "data"),
)
def spin_globe(n, speed, rot_lon, rot_lat):
    # Increment longitude by speed each tick
    rot_lon = (rot_lon + (speed or 0)) % 360
    return make_figure(rot_lon=rot_lon, rot_lat=rot_lat), rot_lon

# Click handler to show details
@app.callback(
    Output("info", "children"),
    Input("globe", "clickData"),
    prevent_initial_call=True
)
def show_click(clickData):
    if not clickData:
        return ""
    pt = clickData["points"][0]
    return f"Clicked: {pt.get('text')} (Lon {pt.get('lon'):.2f}°, Lat {pt.get('lat'):.2f}°)"

if __name__ == "__main__":
    app.run(debug=True)
