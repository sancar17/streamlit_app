import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import os
import numpy as np
import umap
import plotly.graph_objects as go
from tqdm import tqdm
import gdown
import zipfile
import base64
from io import BytesIO

import streamlit.components.v1 as components
import json

import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Spectral10
from bokeh.embed import json_item
import json

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Google Drive URLs 
GDRIVE_URLS = {
    "sample_data": "https://drive.google.com/uc?id=1c-OBD9x_RT_VX0GZUbmOeEIFgpEdNNRH",
    "DinoBloom S": "https://drive.google.com/uc?id=1gedjQGhf4FiYpF1tP40ugMaYc0t6GhZZ",
    "DinoBloom B": "https://drive.google.com/uc?id=1gho7OcsJlekf8Pu84blhVBFT0WoPDolc",
    "DinoBloom L": "https://drive.google.com/uc?id=1L1ahUiQuTlpP2LItYa4JRYJwDMUnFzal",
    "DinoBloom G": "https://drive.google.com/uc?id=16VT6rCL4QY0sUkJ1UmPnLVkCQXD8Fdwi"
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

def download_from_gdrive(gdrive_url, download_path):
    if os.path.dirname(download_path):
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
    gdown.download(gdrive_url, download_path, quiet=False)

# Check if a file exists
def check_if_file_exists(filepath):
    return os.path.isfile(filepath)

# Check if a directory exists
def check_if_directory_exists(dirpath):
    return os.path.isdir(dirpath)

# Function to list files in a directory
def list_files_in_directory(directory, file_extension=None):
    try:
        files = os.listdir(directory)
        if file_extension:
            files = [file for file in files if file.endswith(file_extension)]
        return files
    except FileNotFoundError:
        return f"Directory {directory} not found."
    except Exception as e:
        return str(e)

# Function to remove all .pt and .pth files in a directory
def remove_pt_pth_files(directory):
    removed_files = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pt') or file_name.endswith('.pth'):
            file_path = os.path.join(directory, file_name)
            os.remove(file_path)
            removed_files.append(file_path)
    return removed_files

# Display files in the directories
st.write("Files in /mount/src/streamlit_app:")
st.write(list_files_in_directory("/mount/src/streamlit_app"))

st.write("Files in /mount/src/streamlit_app/sample_data:")
st.write(list_files_in_directory("/mount/src/streamlit_app/sample_data"))

def load_images(data_folder):
    st.write(f"Loading images from {data_folder}")
    image_paths = []
    labels = []
    nested_data_folder = os.path.join(data_folder, "data_sample")
    class_names = os.listdir(nested_data_folder)
    st.write(f"Class names found: {class_names}")
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    for cls_name in class_names:
        cls_folder = os.path.join(nested_data_folder, cls_name)
        if os.path.isdir(cls_folder):
            for image_name in os.listdir(cls_folder):
                if image_name.endswith('.bmp'):
                    image_paths.append(os.path.join(cls_folder, image_name))
                    labels.append(class_to_idx[cls_name])
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    images = []
    valid_labels = []
    valid_image_paths = []
    for image_path, label in tqdm(zip(image_paths, labels), desc="Loading images", total=len(image_paths)):
        try:
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image)
            images.append(image)
            valid_labels.append(label)
            valid_image_paths.append(image_path)
        except (OSError, IOError) as e:
            st.write(f"Skipping corrupted image: {image_path}, error: {e}")
    
    if not images:
        st.write("No valid images found")
    else:
        st.write(f"Found {len(images)} valid images")
    
    images = torch.stack(images)
    labels = np.array(valid_labels)
    return images, labels, class_names, valid_image_paths

def create_interactive_umap_with_images(data, labels, image_paths, class_names):
    reducer = umap.UMAP()
    umap_data = reducer.fit_transform(data)

    # Create a color map
    unique_labels = list(set(labels))
    color_map = {label: Spectral10[i % 10] for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]

    source = ColumnDataSource(data=dict(
        x=umap_data[:, 0],
        y=umap_data[:, 1],
        label=[class_names[label] for label in labels],
        color=colors,
        image_url=image_paths
    ))

    p = figure(title="UMAP Projection with Images", width=800, height=600)
    
    p.scatter('x', 'y', size=10, color='color', alpha=0.6, source=source)

    hover = HoverTool(tooltips="""
        <div>
            <div>
                <img src="@image_url" style="width: 200px; height: 200px;"/>
            </div>
            <div>
                <span style="font-size: 12px; color: #966;">@label</span>
            </div>
        </div>
    """)

    p.add_tools(hover)

    return p

def upload_and_process_features(features_file, data_source, data_file):
    if features_file is not None:
        features = np.load(features_file)
    else:
        raise ValueError("Features file is required for this option.")
    
    data_path = "sample_data"
    if not check_if_directory_exists(data_path):
        st.write("Downloading data sample...")
        download_path = os.path.join(data_path, "sample_data.zip")
        download_from_gdrive(GDRIVE_URLS["sample_data"], download_path)
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    else:
        st.write("Using data sample from the cloud.")

    _, labels, class_names, image_paths = load_images(data_path)
    umap_fig = create_interactive_umap_with_images(features, labels, image_paths, class_names)
    return umap_fig

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

def upload_and_process_data_and_model(model_source, model_file, data_source, data_file):
    model_key = model_source
    model_path = f"{model_key.replace(' ', '_')}.pth"
    if not check_if_file_exists(model_path):
        st.write(f"Downloading model {model_source}...")
        st.write(f'Model Path: {model_path}')
        download_from_gdrive(GDRIVE_URLS[model_key], model_path)
    else:
        st.write(f"Using model {model_source} from the cloud.")
    
    model = get_dino_bloom(model_path, model_options[model_key])

    data_path = "sample_data"
    if not check_if_directory_exists(data_path):
        st.write("Downloading data sample...")
        os.makedirs(data_path, exist_ok=True)  # Ensure the directory exists
        download_path = os.path.join(data_path, "sample_data.zip")
        download_from_gdrive(GDRIVE_URLS["sample_data"], download_path)
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    else:
        st.write("Using data sample from the cloud.")
    
    images, labels, class_names, image_paths = load_images(data_path)
    images = images
    
    with torch.no_grad():
        features = model(images).cpu().numpy()
    
    bokeh_plot = create_interactive_umap_with_images(features, labels, image_paths, class_names)
    return bokeh_plot

# Streamlit app
st.title("UMAP Visualization with DinoBloom Features")

# Button to remove predownloaded models
if st.button("Remove predownloaded models"):
    removed_files = remove_pt_pth_files("/mount/src/streamlit_app")
    st.write("Removed files:")
    st.write(removed_files)

option = st.radio("Choose an option", ["Use Features", "Use Model"])

if option == "Use Features":
    features_file = st.file_uploader("Upload Features File (required)", type=["npy"])
    data_source = st.selectbox("Choose Data Source", ["Sample Data", "Upload Data"])
    if data_source == "Upload Data":
        data_file = st.file_uploader("Upload Data Folder (optional)")
    else:
        data_file = None
    if st.button("Visualize UMAP"):
        if features_file is not None:
            fig = upload_and_process_features(features_file, data_source, data_file)
            st.plotly_chart(fig)
        else:
            st.error("Please upload a features file.")
else:
    model_source = st.selectbox("Choose Model", ["DinoBloom S", "DinoBloom B", "DinoBloom L", "DinoBloom G", "Upload Model"])
    if model_source == "Upload Model":
        model_file = st.file_uploader("Upload Model File (optional)", type=["pth"])
    else:
        model_file = None
    data_source = st.selectbox("Choose Data Source", ["Sample Data", "Upload Data"])
    if data_source == "Upload Data":
        data_file = st.file_uploader("Upload Data Folder (optional)")
    else:
        data_file = None
    if st.button("Visualize UMAP"):
        if model_source != "Upload Model" or model_file is not None:
            bokeh_plot = upload_and_process_data_and_model(model_source, model_file, data_source, data_file)
            st.bokeh_chart(bokeh_plot)
        else:
            st.error("Please select a model or upload a model file.")
