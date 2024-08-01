import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import torch
from pyngrok import ngrok
from PIL import Image
import os
import numpy as np
import umap
import base64
from io import BytesIO
import zipfile
import plotly.express as px
import load
from load import load_images, get_dino_bloom
import traceback

print("Starting the application...")

# Initialize Dash app
app = dash.Dash(__name__)

# Local paths
LOCAL_PATHS = load.LOCAL_PATHS
embed_sizes = load.embed_sizes
model_options = load.model_options

print("Local paths:", LOCAL_PATHS)
print("Model options:", model_options)

# Global variables to store UMAP data and labels
umap_data = None
global_labels = None
global_class_names = None
global_image_paths = None

def create_umap_visualization(umap_data, labels, image_paths, class_names):
    print(f"Creating UMAP visualization with {len(labels)} points")
    # Create a color map
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Plotly[:len(unique_labels)]
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    
    # Assign colors to each point
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
        customdata=image_paths,  # Store image paths in customdata
        hoverinfo="text",
        hovertemplate="%{text}<extra></extra>",
    )

    layout = go.Layout(
        title="UMAP Projection with Images",
        xaxis=dict(title="UMAP 1"),
        yaxis=dict(title="UMAP 2"),
        hovermode='closest',
        dragmode='select'  # Enable rectangle selection
    )

    fig = go.Figure(data=[trace], layout=layout)

    # Add a legend
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
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': k, 'value': k} for k in model_options.keys()],
        value='DinoBloom S'
    ),
    dcc.Graph(id='umap-plot', clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
    html.Div([
        dcc.Input(id='label-input', type='text', placeholder='Enter new label'),
        html.Button('Apply Label', id='apply-label-button', n_clicks=0)
    ]),
    html.Div(id='label-status'),
    dcc.Store(id='is-data-loaded', data=False)
])

@app.callback(
    Output('umap-plot', 'figure'),
    Output('label-status', 'children'),
    Output('is-data-loaded', 'data'),
    Input('model-dropdown', 'value'),
    Input('apply-label-button', 'n_clicks'),
    Input('is-data-loaded', 'data'),
    State('label-input', 'value'),
    State('umap-plot', 'selectedData'),
    State('umap-plot', 'figure')
)
def update_graph_and_labels(selected_model, n_clicks, is_data_loaded, new_label, selected_data, current_figure):
    global umap_data, global_labels, global_class_names, global_image_paths

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not is_data_loaded or trigger_id == 'model-dropdown':
        try:
            print(f"Loading data for model: {selected_model}")
            model_key = selected_model
            model_path = LOCAL_PATHS[model_key]
            
            if not os.path.isfile(model_path):
                print(f"Error: Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Using model {selected_model} from local path: {model_path}")
            model = get_dino_bloom(model_path, model_options[model_key])

            data_path = "sample_data"
            if not os.path.isdir(data_path):
                print("Extracting data sample...")
                with zipfile.ZipFile(LOCAL_PATHS["sample_data"], 'r') as zip_ref:
                    zip_ref.extractall(data_path)
            else:
                print("Using existing data sample.")
            
            images, labels, class_names, image_paths = load_images(data_path)
            print(f"Loaded {len(images)} images, {len(class_names)} classes")
            
            with torch.no_grad():
                features = model(images).cpu().numpy()
            
            print(f"Extracted features shape: {features.shape}")
            
            reducer = umap.UMAP(n_neighbors=min(15, len(features) - 1), min_dist=0.1, metric='cosine')
            umap_data = reducer.fit_transform(features)
            print(f"UMAP data shape: {umap_data.shape}")

            # Initialize all labels as "unlabeled"
            global_labels = np.zeros(len(labels), dtype=int)
            global_class_names = ["unlabeled"]
            global_image_paths = image_paths

            fig = create_umap_visualization(umap_data, global_labels, global_image_paths, global_class_names)
            return fig, "Data loaded and visualized.", True
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
            return go.Figure(), f"Error: {str(e)}", False

    elif trigger_id == 'apply-label-button':
        if not selected_data or not new_label:
            return current_figure, "Please select points and enter a label.", True

        selected_indices = [point['pointIndex'] for point in selected_data['points']]
        
        # Update labels
        if new_label not in global_class_names:
            global_class_names.append(new_label)
        new_label_index = global_class_names.index(new_label)
        global_labels[selected_indices] = new_label_index

        # Update the figure
        updated_figure = create_umap_visualization(umap_data, global_labels, global_image_paths, global_class_names)
        
        return updated_figure, f"Applied label '{new_label}' to {len(selected_indices)} points.", True

    # If neither input was triggered (should not happen due to is_data_loaded)
    return current_figure, "Ready.", True

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

    # Load and encode image
    im = Image.open(image_path)
    im.thumbnail((200, 200))  # Resize image
    buffer = BytesIO()
    im.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/png;base64, " + encoded_image

    children = [
        html.Img(src=im_url, style={"width": "200px"}),
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