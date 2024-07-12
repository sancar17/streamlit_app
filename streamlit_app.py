import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import os
import numpy as np
import umap
import plotly.express as px
from tqdm import tqdm
import gdown

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to download file from Google Drive
def download_from_gdrive(gdrive_url, download_path):
    gdown.download(gdrive_url, download_path, quiet=False)

DEFAULT_MODEL_PATH = "models/"
DEFAULT_DATA_PATH = "data/"
GDRIVE_URLS = {
    "sample_data": st.secrets["GDRIVE_URL_SAMPLE_DATA"],
    "DinoBloom S": st.secrets["GDRIVE_URL_DINOBLOOM_S"],
    "DinoBloom B": st.secrets["GDRIVE_URL_DINOBLOOM_B"],
    "DinoBloom L": st.secrets["GDRIVE_URL_DINOBLOOM_L"],
    "DinoBloom G": st.secrets["GDRIVE_URL_DINOBLOOM_G"]
}

def load_images(data_folder):
    image_paths = []
    labels = []
    class_names = os.listdir(data_folder)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    for cls_name in class_names:
        cls_folder = os.path.join(data_folder, cls_name)
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
            print(f"Skipping corrupted image: {image_path}, error: {e}")
    
    images = torch.stack(images)
    labels = np.array(valid_labels)
    return images, labels, class_names, valid_image_paths

def create_interactive_umap_with_images(data, labels, image_paths, class_names):
    reducer = umap.UMAP()
    umap_data = reducer.fit_transform(data)
    
    small_images = []
    for image_path in image_paths:
        image = Image.open(image_path).resize((2, 2)).convert('RGB')
        small_images.append(np.array(image))

    fig = px.scatter(
        umap_data, x=0, y=1, color=labels, labels={'color': 'Class'},
        hover_data={'Class Name': [class_names[label] for label in labels]}
    )
    
    for trace in fig.data:
        trace.marker.size = 10

    fig.update_layout(
        title="Interactive UMAP Projection with Images",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    return fig

def upload_and_process_features(features_file, data_source, data_file):
    if features_file is not None:
        features = np.load(features_file)
    else:
        raise ValueError("Features file is required for this option.")
    
    if data_source == "Sample Data":
        data_path = os.path.join(DEFAULT_DATA_PATH, "sample_data")
        download_from_gdrive(GDRIVE_URLS["sample_data"], data_path)
    elif data_file is not None:
        data_path = os.path.dirname(data_file.name)
    else:
        raise ValueError("Data source is required for this option.")

    _, labels, class_names, image_paths = load_images(data_path)
    umap_fig = create_interactive_umap_with_images(features, labels, image_paths, class_names)
    return umap_fig

def upload_and_process_data_and_model(model_source, model_file, data_source, data_file):
    if model_source != "Upload Model":
        model_key = model_source
        model_path = os.path.join(DEFAULT_MODEL_PATH, f"{model_key.replace(' ', '_')}.pth")
        download_from_gdrive(GDRIVE_URLS[model_key], model_path)
    elif model_file is not None:
        model_path = model_file.name
    else:
        raise ValueError("Model source is required for this option.")

    if data_source == "Sample Data":
        data_path = os.path.join(DEFAULT_DATA_PATH, "sample_data")
        download_from_gdrive(GDRIVE_URLS["sample_data"], data_path)
    elif data_file is not None:
        data_path = os.path.dirname(data_file.name)
    else:
        raise ValueError("Data source is required for this option.")
    
    model = torch.load(model_path)  # Directly load the model from the path
    images, labels, class_names, image_paths = load_images(data_path)
    images = images.cuda()
    
    with torch.no_grad():
        features = model(images).cpu().numpy()
    
    umap_fig = create_interactive_umap_with_images(features, labels, image_paths, class_names)
    return umap_fig

st.title("UMAP Visualization with DINOv2 Features")
option = st.radio("Choose an option", ["Use Features", "Use Model"])

if option == "Use Features":
    features_file = st.file_uploader("Upload Features File (required)", type=["npy"])
    data_source = st.selectbox("Choose Data Source", ["Sample Data", "Upload Data"])
    data_file = st.file_uploader("Upload Data Folder (optional)")
    if st.button("Visualize UMAP"):
        if features_file is not None:
            fig = upload_and_process_features(features_file, data_source, data_file)
            st.plotly_chart(fig)
        else:
            st.error("Please upload a features file.")
else:
    model_source = st.selectbox("Choose Model", ["DinoBloom S", "DinoBloom B", "DinoBloom L", "DinoBloom G", "Upload Model"])
    model_file = st.file_uploader("Upload Model File (optional)", type=["pth"])
    data_source = st.selectbox("Choose Data Source", ["Sample Data", "Upload Data"])
    data_file = st.file_uploader("Upload Data Folder (optional)")
    if st.button("Visualize UMAP"):
        if model_source != "Upload Model" or model_file is not None:
            fig = upload_and_process_data_and_model(model_source, model_file, data_source, data_file)
            st.plotly_chart(fig)
        else:
            st.error("Please select a model or upload a model file.")
