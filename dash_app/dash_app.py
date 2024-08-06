import dash
from dash import dcc, html, no_update, clientside_callback
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import torch
import numpy as np
import umap
import base64
import plotly.express as px
import load
from load import load_images, get_dino_bloom
import traceback
import io
from pyngrok import ngrok
import zipfile
from flask_caching import Cache
import time

print("Starting the application...")

# Initialize Dash app
app = dash.Dash(__name__)

# Setup cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Local model paths
LOCAL_PATHS = load.LOCAL_PATHS
model_options = load.model_options

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables
global_data = {}

def create_umap_visualization(umap_data, labels, image_paths, class_names):
    print(f"Creating UMAP visualization with {len(labels)} points")
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Plotly[:len(unique_labels)]
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    
    point_colors = [color_map[label] for label in labels]

    trace = go.Scatter(
        x=umap_data[:, 0],
        y=umap_data[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=point_colors,
            opacity=0.8,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=[class_names[label] if label < len(class_names) else f"Unknown Label {label}" for label in labels],
        customdata=image_paths,
        hoverinfo="text",
        hovertemplate="%{text}<extra></extra>",
    )

    layout = go.Layout(
        title="UMAP Projection with Images",
        xaxis=dict(title="UMAP 1"),
        yaxis=dict(title="UMAP 2"),
        hovermode='closest',
        dragmode='select'
    )

    fig = go.Figure(data=[trace], layout=layout)

    for label, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=class_names[label] if label < len(class_names) else f"Unknown Label {label}",
            showlegend=True,
            name=class_names[label] if label < len(class_names) else f"Unknown Label {label}"
        ))

    return fig

# App layout
app.layout = html.Div([
    html.H1("UMAP Visualization with DinoBloom Features"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Data (Zip)'),
        multiple=False,
        max_size=-1
    ),
    html.Div(id='upload-status'),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': k, 'value': k} for k in model_options.keys()],
        value='select'
    ),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=[dcc.Graph(id='umap-plot', clear_on_unhover=True)]
    ),
    dcc.Tooltip(id="graph-tooltip"),
    html.Div([
        dcc.Input(id='label-input', type='text', placeholder='Enter new label'),
        html.Button('Apply Label', id='apply-label-button', n_clicks=0)
    ]),
    html.Div(id='label-status'),
    dcc.Store(id='is-data-loaded', data=False),
    dcc.Store(id='uploaded-data', data=None),
    dcc.Store(id='upload-status-store', data=''),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

# New client-side callback for upload status
clientside_callback(
    """
    function(contents, filename) {
        if (contents) {
            return 'Uploading ' + filename + '...';
        }
        return '';
    }
    """,
    Output('upload-status-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

@app.callback(
    [Output('uploaded-data', 'data'),
     Output('is-data-loaded', 'data'),
     Output('upload-status', 'children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')]
)
def process_upload(contents, filename):
    if contents is None:
        return no_update, False, "Waiting for file..."
    
    try:
        # Handle different content formats
        if isinstance(contents, str):
            if ',' in contents:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
            else:
                decoded = base64.b64decode(contents)
        elif isinstance(contents, bytes):
            decoded = contents
        else:
            raise ValueError(f"Unsupported content format: {type(contents)}")
        
        print(f"Decoded data type: {type(decoded)}")
        print(f"Decoded data length: {len(decoded)} bytes")
        
        if len(decoded) == 0:
            raise ValueError("The uploaded file is empty.")
        
        zip_file = io.BytesIO(decoded)
        
        if not zipfile.is_zipfile(zip_file):
            raise zipfile.BadZipFile("The uploaded file is not a valid ZIP file.")
        
        images, labels, class_names, image_paths, valid_image_paths = load_images(zip_file)
        print(f"Loaded {len(images)} images, {len(class_names)} classes")
        
        if len(images) == 0:
            raise ValueError("No valid images found in the ZIP file.")
        
        global_data['images'] = images
        global_data['labels'] = labels
        global_data['class_names'] = class_names
        global_data['image_paths'] = image_paths
        global_data['valid_image_paths'] = valid_image_paths
        
        return {'status': 'success'}, True, f"Uploaded {filename} successfully. {len(images)} images processed."

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        return no_update, False, f"Upload failed: {str(e)}"

@cache.memoize()
def get_model(model_key):
    model_path = LOCAL_PATHS[model_key]
    return get_dino_bloom(model_path, load.model_options[model_key])

@app.callback(
    [Output('umap-plot', 'figure'),
     Output('label-status', 'children')],
    [Input('model-dropdown', 'value'),
     Input('is-data-loaded', 'data'),
     Input('apply-label-button', 'n_clicks')],
    [State('label-input', 'value'),
     State('umap-plot', 'selectedData')]
)
def update_graph_and_apply_label(selected_model, is_data_loaded, n_clicks, new_label, selected_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not is_data_loaded:
        return go.Figure(), "No data loaded"

    if triggered_id == 'model-dropdown':
        if selected_model == 'select':
            return go.Figure(), "Please select a model"
        
        try:
            model = get_model(selected_model)
            
            images = global_data['images'].to(device)
            
            with torch.no_grad():
                features = model(images).cpu().numpy()
            
            reducer = umap.UMAP(n_neighbors=min(15, len(features) - 1), min_dist=0.1, metric='cosine')
            umap_data = reducer.fit_transform(features)
            
            global_data['umap_data'] = umap_data
            
            fig = create_umap_visualization(umap_data, global_data['labels'], global_data['image_paths'], global_data['class_names'])
            return fig, "UMAP visualization created"
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
            return go.Figure(), f"Error: {str(e)}"

    elif triggered_id == 'apply-label-button':
        if not n_clicks or not new_label or not selected_data:
            return no_update, "No label applied"
        
        selected_indices = [point['pointIndex'] for point in selected_data['points']]
        
        if new_label not in global_data['class_names']:
            global_data['class_names'].append(new_label)
        new_label_index = global_data['class_names'].index(new_label)
        global_data['labels'][selected_indices] = new_label_index

        updated_figure = create_umap_visualization(global_data['umap_data'], global_data['labels'], global_data['image_paths'], global_data['class_names'])
        
        return updated_figure, f"Applied label '{new_label}' to {len(selected_indices)} points."

    return go.Figure(), "Ready"

@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("umap-plot", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointIndex"]

    image_path = pt["customdata"]
    class_name = pt["text"]

    children = [
        html.Img(src=f"data:image/png;base64,{image_path}", style={"width": "200px"}),
        html.P(f"{class_name}", style={"color": "darkblue"})
    ]

    return True, bbox, children

if __name__ == '__main__':
    print("Starting the Dash server...")
    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(8050)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:8050\"")

    # Update the server name
    app.run_server(debug=True, use_reloader=False)

    # Close the ngrok tunnel
    ngrok.kill()