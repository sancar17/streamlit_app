import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import torch
import numpy as np
import umap
import base64
from io import BytesIO
import zipfile
import plotly.express as px
import load
from load import load_images, get_dino_bloom
import traceback
import io
from pyngrok import ngrok
import io
import base64
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn


print("Starting the application...")

# Initialize Dash app
app = dash.Dash(__name__)

# Local model paths
LOCAL_PATHS = {
    "DinoBloom S": "/home/ubuntu/dino-vis/models/DinoBloom-S.pth",
    "DinoBloom B": "/home/ubuntu/dino-vis/models/DinoBloom-B.pth",
    "DinoBloom L": "/home/ubuntu/dino-vis/models/DinoBloom-L.pth",
    "DinoBloom G": "/home/ubuntu/dino-vis/models/DinoBloom-G.pth"
}

model_options = load.model_options

# Global variables to store UMAP data and labels
umap_data = None
global_labels = None
global_class_names = None
global_image_paths = None

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
        multiple=False
    ),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': k, 'value': k} for k in model_options.keys()],
        value='select'
    ),
    dcc.Graph(id='umap-plot', clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
    html.Div([
        dcc.Input(id='label-input', type='text', placeholder='Enter new label'),
        html.Button('Apply Label', id='apply-label-button', n_clicks=0)
    ]),
    html.Div(id='label-status'),
    dcc.Store(id='is-data-loaded', data=False),
    dcc.Store(id='uploaded-data', data=None)
])

def convert_tensor_to_base64_image(tensor_image):
    # Convert tensor to PIL Image
    pil_image = transforms.ToPILImage()(tensor_image)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.callback(
    [Output('uploaded-data', 'data'),
     Output('is-data-loaded', 'data'),
     Output('umap-plot', 'figure'),
     Output('label-status', 'children')],
    [Input('upload-data', 'contents'),
     Input('model-dropdown', 'value'),
     Input('apply-label-button', 'n_clicks')],
    [State('label-input', 'value'),
     State('umap-plot', 'selectedData'),
     State('uploaded-data', 'data')]
)
def update_graph_and_labels(uploaded_file, selected_model, n_clicks, new_label, selected_data, uploaded_data_state):
    global umap_data, global_labels, global_class_names, global_image_paths

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'upload-data':
        if uploaded_file is None:
            return no_update, False, no_update, "No data uploaded."
        
        content_type, content_string = uploaded_file.split(',')
        decoded = base64.b64decode(content_string)
        zip_file = io.BytesIO(decoded)
        
        try:
            images, labels, class_names, image_paths = load_images(zip_file)
            print(f"Loaded {len(images)} images, {len(class_names)} classes")
            
            # Convert images to base64 encoded strings
            image_base64_list = [convert_tensor_to_base64_image(img) for img in images]
            
            return {
                'images': image_base64_list,
                'labels': labels.tolist(),
                'class_names': class_names
            }, True, no_update, "Data uploaded successfully."
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
            return no_update, False, no_update, f"Error: {str(e)}"

    if trigger_id == 'model-dropdown':
        if not uploaded_data_state:
            return no_update, no_update, no_update, "Please upload data first."
        
        try:
            model_key = selected_model
            model_path = LOCAL_PATHS[model_key]
            model = get_dino_bloom(model_path, load.model_options[model_key])
            
            images = [transforms.ToTensor()(Image.open(io.BytesIO(base64.b64decode(img)))) for img in uploaded_data_state['images']]
            images = torch.stack(images)
            
            with torch.no_grad():
                features = model(images).cpu().numpy()
            
            reducer = umap.UMAP(n_neighbors=min(15, len(features) - 1), min_dist=0.1, metric='cosine')
            umap_data = reducer.fit_transform(features)
            
            global_labels = np.zeros(len(uploaded_data_state['labels']), dtype=int)
            global_class_names = uploaded_data_state['class_names']
            global_image_paths = uploaded_data_state['images']

            fig = create_umap_visualization(umap_data, global_labels, global_image_paths, global_class_names)
            return uploaded_data_state, True, fig, "Data visualized."
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
            return uploaded_data_state, False, no_update, f"Error: {str(e)}"

    if trigger_id == 'apply-label-button':
        if not selected_data or not new_label:
            return no_update, no_update, no_update, "Please select points and enter a label."
        
        selected_indices = [point['pointIndex'] for point in selected_data['points']]
        
        if new_label not in global_class_names:
            global_class_names.append(new_label)
        new_label_index = global_class_names.index(new_label)
        global_labels[selected_indices] = new_label_index

        updated_figure = create_umap_visualization(umap_data, global_labels, global_image_paths, global_class_names)
        
        return uploaded_data_state, True, updated_figure, f"Applied label '{new_label}' to {len(selected_indices)} points."

    return no_update, no_update, no_update, "Ready."

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

    im = Image.open(io.BytesIO(base64.b64decode(image_path)))
    im.thumbnail((200, 200))
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
