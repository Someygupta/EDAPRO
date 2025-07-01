import dash
import os
from dash import dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import io
import base64
import plotly.express as px
import numpy as np
import soundfile as sf
import plotly.graph_objs as go
from sqlalchemy import create_engine
import pymysql

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

def login_layout():
    return html.Div(
        style={
            'backgroundImage': 'url("/assets/background.jpg")',
            'backgroundSize': 'cover',
            'backgroundPosition': 'center',
            'height': '100vh',
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'flexDirection': 'column'
        },
        children=[
            html.Div([
                html.H2("🔐 Login to EDAPRO", style={
                    'color': '#00ffff',
                    'textAlign': 'center',
                    'marginBottom': '20px',
                    'textShadow': '0 0 10px #00ffff'
                }),
                dbc.Input(id="username", placeholder="Username", type="text", className="mb-2", style={
                    "width": "250px", "margin": "auto"
                }),
                dbc.Input(id="password", placeholder="Password", type="password", className="mb-3", style={
                    "width": "250px", "margin": "auto"
                }),
                dbc.Button("Login", id="login-button", color="info", style={
                    "width": "250px", "margin": "auto", "fontWeight": "bold"
                }),
                html.Div(id="login-alert", className="text-danger mt-2", style={'textAlign': 'center'})
            ], style={
                'backgroundColor': 'rgba(0, 0, 0, 0.7)',
                'padding': '40px',
                'borderRadius': '20px',
                'boxShadow': '0 0 15px cyan',
                'textAlign': 'center'
            })
        ]
    )

def dashboard_layout():
    return dbc.Container([
        html.H2("📊 AutoEDA Dashboard", className="text-center my-3"),

        dcc.Upload(id='upload-data', children=dbc.Button("📁 Upload File", color="secondary", className="mb-2")),
        html.Div(id="output-data-upload", className="text-success mb-3"),
        html.Div(id="sheet-selector-container"),

        html.H5("🛢️ Load from MySQL Database:"),
        dbc.Row([
            dbc.Col(dbc.Input(id='sql-host', placeholder='Host', value='localhost')),
            dbc.Col(dbc.Input(id='sql-port', placeholder='Port', value='3306')),
            dbc.Col(dbc.Input(id='sql-user', placeholder='User', value='root')),
            dbc.Col(dbc.Input(id='sql-pass', placeholder='Password', type='password')),
            dbc.Col(dbc.Input(id='sql-db', placeholder='Database')),
            dbc.Col(dbc.Input(id='sql-query', placeholder='Table or SELECT query')),
            dbc.Col(dbc.Button("Load SQL", id='load-sql', color="info"))
        ], className="mb-3"),
        html.Div(id='sql-upload-status', className="text-success mb-2"),
        dcc.Store(id='sql-data-store'),

        dbc.Row([
            dbc.Col([
                html.H5("✅ Select EDA Options:"),
                dbc.Checklist(
                    id='eda-options',
                    options=[
                        {'label': 'Basic Statistics ℹ️', 'value': 'stats'},
                        {'label': 'Missing Values 🔍', 'value': 'missing'},
                        {'label': 'Outliers 🚨', 'value': 'outliers'},
                        {'label': 'Correlation Matrix 📈', 'value': 'correlation'},
                        {'label': 'Column-wise Plot 📊', 'value': 'column_plot'}
                    ],
                    value=['stats', 'missing'],
                    inline=False,
                    className="mb-3"
                )
            ], width=6),
        ]),

        dcc.Dropdown(id='plot-column-dropdown', placeholder="Select a column to plot", className="mb-3"),

        dbc.Button("Generate Insights", id='generate-button', color="success", className="mb-3"),
        dcc.Loading(html.Div(id='eda-output'), type="default"),
        html.Div(id="download-links", className="my-3"),

        html.Hr(),
        html.H5("🔊 Upload Audio"),
        dcc.Upload(id='upload-audio', children=dbc.Button("🎵 Upload Audio File", color="info", className="mb-2")),
        html.Div(id="audio-file-name", className="text-success mb-2"),
        html.Div(id="audio-waveform-output")
    ], fluid=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    return login_layout() if pathname in ['/', '/login'] else dashboard_layout()

@app.callback(
    Output("url", "pathname"),
    Output("login-alert", "children"),
    Input("login-button", "n_clicks"),
    State("username", "value"),
    State("password", "value"),
    State("url", "pathname"),
    prevent_initial_call=True
)
def login(n_clicks, user, pwd, current_path):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    if user == "admin" and pwd == "admin@2025":
        return "/dashboard", ""
    return "/login", "❌ Invalid username or password."

@app.callback(
    Output('sheet-selector-container', 'children'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_sheet_selector(contents, filename):
    if contents:
        msg = f"✅ File '{filename}' uploaded successfully."
        if filename.endswith('.xlsx'):
            decoded = base64.b64decode(contents.split(',')[1])
            sheet_names = pd.ExcelFile(io.BytesIO(decoded)).sheet_names
            return dcc.Dropdown(id='sheet-name', options=[{'label': s, 'value': s} for s in sheet_names], value=sheet_names[0]), msg
        return None, msg
    return None, ""

@app.callback(
    Output('sql-upload-status', 'children'),
    Output('sql-data-store', 'data'),
    Input('load-sql', 'n_clicks'),
    State('sql-host', 'value'),
    State('sql-port', 'value'),
    State('sql-user', 'value'),
    State('sql-pass', 'value'),
    State('sql-db', 'value'),
    State('sql-query', 'value'),
    prevent_initial_call=True
)
def load_sql_data(n_clicks, host, port, user, pwd, db, query):
    try:
        engine = create_engine(f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}")
        if not query.lower().strip().startswith("select"):
            query = f"SELECT * FROM {query}"
        df = pd.read_sql(query, con=engine)
        return f"✅ Loaded {len(df)} rows from MySQL.", df.to_json(date_format='iso', orient='split')
    except Exception as e:
        return f"❌ SQL Error: {str(e)}", None

@app.callback(
    Output('eda-output', 'children'),
    Output('plot-column-dropdown', 'options'),
    Output('download-links', 'children'),
    Input('generate-button', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('eda-options', 'value'),
    State('plot-column-dropdown', 'value'),
    State('sheet-selector-container', 'children'),
    State('sql-data-store', 'data'),
    prevent_initial_call=True
)
def generate_eda(n, contents, filename, selected_options, plot_col, sheet_selector, sql_data):
    if sql_data:
        df = pd.read_json(sql_data, orient='split')
    elif contents:
        sheet_name = None
        if sheet_selector and isinstance(sheet_selector, dict):
            props = sheet_selector.get("props", {})
            sheet_name = props.get("value")
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.xlsx'):
            xls = pd.ExcelFile(io.BytesIO(decoded))
            df = xls.parse(sheet_name or xls.sheet_names[0])
        else:
            return "Unsupported file type.", [], ""
    else:
        return "Please upload a file or load from SQL.", [], ""

    output = []
    downloads = []
    col_options = [{"label": col, "value": col} for col in df.columns if df[col].dtype in ['int64', 'float64']]

    try:
        if 'stats' in selected_options:
            stats = df.describe().reset_index()
            output.append(html.H5("📊 Basic Statistics"))
            output.append(dash_table.DataTable(stats.to_dict('records'), columns=[{"name": i, "id": i} for i in stats.columns]))
            downloads.append(html.A("📥 Download Stats", href="data:text/csv;charset=utf-8," + stats.to_csv(index=False), download="stats.csv", target="_blank"))

        if 'missing' in selected_options:
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Column", "MissingCount"]
            output.append(html.H5("🔍 Missing Values"))
            output.append(dash_table.DataTable(missing.to_dict('records'), columns=[{"name": i, "id": i} for i in missing.columns]))
            downloads.append(html.A("📥 Download Missing", href="data:text/csv;charset=utf-8," + missing.to_csv(index=False), download="missing.csv", target="_blank"))

        if 'outliers' in selected_options:
            num_df = df.select_dtypes(include=['number'])
            outlier_counts = num_df.apply(lambda x: ((x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75)-x.quantile(0.25)))) | (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75)-x.quantile(0.25))))).sum())
            outliers_df = outlier_counts.reset_index()
            outliers_df.columns = ["Column", "OutlierCount"]
            output.append(html.H5("🚨 Outliers"))
            output.append(dash_table.DataTable(outliers_df.to_dict('records'), columns=[{"name": i, "id": i} for i in outliers_df.columns]))
            downloads.append(html.A("📥 Download Outliers", href="data:text/csv;charset=utf-8," + outliers_df.to_csv(index=False), download="outliers.csv", target="_blank"))

        if 'correlation' in selected_options:
            corr = df.select_dtypes(include=['number']).corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title="📈 Correlation Matrix")
            output.append(dcc.Graph(figure=fig))

        if 'column_plot' in selected_options and plot_col:
            fig = px.histogram(df, x=plot_col, nbins=20, title=f"📊 Distribution of {plot_col}")
            output.append(dcc.Graph(figure=fig))

        return output, col_options, downloads

    except Exception as e:
        return f"❌ Error generating EDA: {str(e)}", col_options, []

@app.callback(
    Output('audio-file-name', 'children'),
    Output('audio-waveform-output', 'children'),
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename')
)
def display_waveform(audio_content, filename):
    if not audio_content:
        return "", ""

    content_type, content_string = audio_content.split(',')
    decoded = base64.b64decode(content_string)
    audio_data, samplerate = sf.read(io.BytesIO(decoded))
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    x = np.linspace(0, len(audio_data) / samplerate, num=len(audio_data))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=audio_data, mode='lines', name='Waveform'))
    fig.update_layout(title="🎧 Audio Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_dark")
    return f"✅ Audio file '{filename}' uploaded successfully.", dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run_server(debug=True)
