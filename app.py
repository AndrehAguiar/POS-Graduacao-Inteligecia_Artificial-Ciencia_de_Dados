# import dash componentes
# dash imports
# image convert
import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
# scientific
import numpy as np
import pandas as pd
# visualização
import plotly.graph_objs as go
from PIL import Image
from dash.dependencies import Input, Output, State
# plotly
from plotly import express as px
from sklearn.cluster import KMeans

df = pd.read_csv("../data/gapminderDataFiveYear.csv")

# stylesheet
# external_stylesheets = ["assets/bWLwgP.assets"]

# intanciar o server
app = dash.Dash(__name__)

# definindo cores
colors = {'background': '#282b38',
          'text': '#a5b1cd'}

# leitura dos dados
# criar um layout
graph = ''

# graph options
graph_options = [{'label': 'População', 'value': 'pop'},
                 {'label': 'Expectativa de Vida', 'value': 'lifeExp'},
                 {'label': 'GDP Per Capita', 'value': 'gdpPercap'}]

# categ options
categ_options = [{'label': 'País', 'value': 'country'},
                 {'label': 'Continente', 'value': 'continent'}]

# categ options
type_options = [{'label': 'Barras', 'value': 'bar'},
                {'label': 'Markers', 'value': 'scatter'}]

# slider options
slider_min, slider_max = df['year'].min(), df['year'].max()
slider_step = 5

# layout
app.layout = html.Div([
    html.H1(children='Hello Dash'),

    html.H6(children='Dash: Um framework de visualização de dados em Python'),

    html.Div([
        dcc.Dropdown(id='id-column',
                     options=graph_options,
                     value=graph_options[0]['value']),
        dcc.Dropdown(id='id-categ',
                     options=categ_options,
                     value=categ_options[0]['value']),
        dcc.Dropdown(id='id-graph',
                     options=type_options,
                     value=type_options[0]['value']),
        html.Button(id='submit-options',
                    n_clicks=0,
                    children='OK')
    ], id='graphics_menu'),

    html.Div(dcc.Loading(dcc.Graph(figure=graph, id='output-div'))),
    html.Hr(),

    html.Div(dcc.Loading([
        html.H2(children="Graph Picker",
                style={'textAlign': 'center'}),
        dcc.Graph(id='graph_picker'),
        dcc.Slider(id='year-picker',
                   min=slider_min,
                   max=slider_max,
                   step=slider_step,
                   marks={int(year): str(year) for year in df['year'].unique()},
                   value=slider_min)
    ])),
    html.Hr(),

    html.H2(children="KMeans segmentation"),
    html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select Image')])),
        dcc.Dropdown(id='choose-k',
                     options=[{'label': str(i), 'value': i} for i in range(1, 11)],
                     value=1),
        html.Button(id='submit-button',
                    n_clicks=0,
                    children='Submit')
    ], id='kmeans_options'),

    html.Div([
        html.Div([
            dcc.Loading(html.Div(id='output-image-upload-raw'))
        ], id='raw_image'),
        html.Div([
            dcc.Loading(html.Div(id='output-image-upload-kmeans'))
        ], id='seg_image')
    ], id='images_container')
], id='content')


# implementando o callback
@app.callback(Output(component_id='output-div', component_property='figure'),
              [Input(component_id='submit-options', component_property='n_clicks')],
              [State('id-column', 'value'),
               State('id-categ', 'value'),
               State('id-graph', 'value')])
def update_output(_, option, categ, type):
    traces = []
    for categoria in df[categ].unique():
        traces.append({'x': df['year'][df[categ] == categoria],
                       'y': df[option][df[categ] == categoria],
                       'type': type, 'name': categoria,
                       'text': df[df[categ] == categ].country})

    graph_fig = {'data': traces,
                 'layout': {'title': f'{option} x {categ} <br /> Gapminder Data Five Year',
                            'plot_bgcolor': colors['background'],
                            'paper_bgcolor': colors['background'],
                            'font': {'color': colors['text']}}}
    return graph_fig


# implementando o callback da figura
@app.callback(Output(component_id='graph_picker', component_property='figure'),
              [Input(component_id='year-picker', component_property='value')])
def update_figure(selected_year):
    # slice no ano correspondente
    filtered_df = df[df['year'] == selected_year]

    # traces
    traces = []
    for country in filtered_df['country'].unique():
        df_by_country = filtered_df[filtered_df['country'] == country]
        traces.append(go.Scatter(
            x=df_by_country['gdpPercap'],
            y=df_by_country['lifeExp'],
            text=df_by_country['country'],
            mode='markers',
            opacity=0.7,
            marker={'size': 15},
            name=country
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            yaxis={'title': 'Life Expectancy'},
            title='GDP Per Capita x Life Expectancy',
            hovermode='closest',
            font={'color': colors['text']},
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background']
        )
    }


@app.callback([Output('output-image-upload-raw', 'children'),
               Output('output-image-upload-kmeans', 'children')],
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('choose-k', 'value')])
def segment_image(_, image, k: int):
    # Callback that applies image segmentation and displays result.
    # convert raw input image to RGB numpy
    img = string_to_rgb(image)
    raw_fig = px.imshow(img)
    raw_fig.update_xaxes(showticklabels=False)
    raw_fig.update_yaxes(showticklabels=False)
    raw_fig.update_traces(hovertemplate=None, hoverinfo='skip')
    raw_fig.update_layout(plot_bgcolor=colors['background'],
                          paper_bgcolor=colors['background'])

    # display raw image
    raw_plot = html.Div([html.H3("Raw image: "),
                         dcc.Graph(figure=raw_fig)])

    # segmented image
    img_kmeans = cluster_image(img, k)

    # kmeans fig
    seg_fig = px.imshow(img_kmeans)
    seg_fig.update_xaxes(showticklabels=False)
    seg_fig.update_yaxes(showticklabels=False)
    seg_fig.update_traces(hovertemplate=None, hoverinfo='skip')
    seg_fig.update_layout(plot_bgcolor=colors['background'],
                          paper_bgcolor=colors['background'])

    # display segmented image
    kmeans_plot = html.Div([html.H3("Segmented image: "),
                            dcc.Graph(figure=seg_fig)])

    return raw_plot, kmeans_plot


def cluster_image(img: np.ndarray, k: int):
    # Apply kmeans to image.
    # reshape to (HxW)x3
    X = img.reshape((-1, 3))
    X = np.float32(X)

    # fit kmeans
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0, n_jobs=-1)
    kmeans.fit(X)

    # parameters
    label, center = kmeans.labels_, kmeans.cluster_centers_

    # now convert back uint8 and reshape to original format
    center = np.uint8(center)

    # pixel's intensity = centroid intensity
    img_kmeans = center[label.flatten()]
    img_kmeans = img_kmeans.reshape((img.shape))

    return img_kmeans


def string_to_rgb(base64_string):
    url = base64_string.split(',')
    image = Image.open(io.BytesIO(base64.b64decode(url[-1])))
    image = image.convert('RGB')
    return np.array(image)


if __name__ == '__main__':
    app.run_server(debug=False, threaded=True)
