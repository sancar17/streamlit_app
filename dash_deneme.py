import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import umap
import base64
from io import BytesIO
import zipfile


# Initialize Dash app
app = dash.Dash(__name__)

# Local paths
LOCAL_PATHS = {
    "sample_data": "/home/ubuntu/data_sample_small.zip",
    "DinoBloom S": "/home/ubuntu/dino-vis/models/DinoBloom-S.pth",
    "DinoBloom B": "/home/ubuntu/dino-vis/models/DinoBloom-B.pth",
    "DinoBloom L": "/home/ubuntu/dino-vis/models/DinoBloom-L.pth",
    "DinoBloom G": "/home/ubuntu/dino-vis/models/DinoBloom-G.pth"
}

embed_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536
}

model_options = {
    "DinoBloom S": "dinov2_vits14",
    "DinoBloom B": "dinov2_vitb14",
    "DinoBloom L": "dinov2_vitl14",
    "DinoBloom G": "dinov2_vitg14"
}

# Helper functions
def check_if_file_exists(filepath):
    return os.path.isfile(filepath)

def check_if_directory_exists(dirpath):
    return os.path.isdir(dirpath)

def load_images(data_folder):
    image_paths = []
    labels = []
    nested_data_folder = os.path.join(data_folder, "data_sample")
    class_names = os.listdir(nested_data_folder)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    for cls_name in class_names:
        cls_folder = os.path.join(nested_data_folder, cls_name)
        if os.path.isdir(cls_folder):
            for image_name in os.listdir(cls_folder):
                if image_name.endswith('.bmp'):
                    image_paths.append(os.path.join(cls_folder, image_name))
                    labels.append(class_to_idx[cls_name])
                    break  # Only take one image from each subfolder
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    images = []
    valid_labels = []
    valid_image_paths = []
    for image_path, label in zip(image_paths, labels):
        try:
            image = Image.open(image_path).convert('RGB')
            images.append(preprocess(image))
            valid_labels.append(label)
            valid_image_paths.append(image_path)
        except (OSError, IOError) as e:
            print(f"Skipping corrupted image: {image_path}, error: {e}")
    
    images = torch.stack(images)
    labels = np.array(valid_labels)
    return images, labels, class_names, valid_image_paths

def get_dino_bloom(modelpath, modelname="dinov2_vitb14"):
    pretrained = torch.load(modelpath, map_location=torch.device('cpu'))
    model = torch.hub.load('facebookresearch/dinov2', modelname)
    
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    pos_embed = nn.Parameter(torch.zeros(1, 257, embed_sizes[modelname]))
    model.pos_embed = pos_embed

    model.load_state_dict(new_state_dict, strict=True)
    return model

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import umap
import base64
from io import BytesIO
import zipfile

# Initialize Dash app
app = dash.Dash(__name__)

# Local paths and other constants remain the same
# ...

def create_umap_visualization(features, labels, image_paths, class_names):
    n_neighbors = min(15, len(features) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric='cosine')
    umap_data = reducer.fit_transform(features)

    trace = go.Scatter(
        x=umap_data[:, 0],
        y=umap_data[:, 1],
        mode='markers',
        marker=dict(size=10, opacity=0.5),
        text=[f"{class_names[label]}" for label in labels],
        customdata=image_paths,  # Store image paths in customdata
        hoverinfo="none",
        hovertemplate=None,
    )

    layout = go.Layout(
        title="UMAP Projection with Images",
        xaxis=dict(title="UMAP 1"),
        yaxis=dict(title="UMAP 2"),
        hovermode='closest'
    )

    fig = go.Figure(data=[trace], layout=layout)
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
    dcc.Tooltip(id="graph-tooltip")
])

# Callback to update the graph
@app.callback(
    Output('umap-plot', 'figure'),
    Input('model-dropdown', 'value')
)
def update_graph(selected_model):
    model_key = selected_model
    model_path = LOCAL_PATHS[model_key]
    
    if not os.path.isfile(model_path):
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
    
    with torch.no_grad():
        features = model(images).cpu().numpy()
    
    fig = create_umap_visualization(features, labels, image_paths, class_names)
    return fig

# Callback for tooltip
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)