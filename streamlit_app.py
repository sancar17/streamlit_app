import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import os
import numpy as np
import umap
import plotly.express as px
from tqdm import tqdm

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def get_dino_bloom(modelpath, modelname="dinov2_vitb14"):
    pretrained = torch.load(modelpath, map_location=torch.device('cuda:0'))
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
    model = model.cuda()
    return model

DEFAULT_MODEL_PATH = "models/DinoBloom-S.pth"
DEFAULT_DATA_PATH = "data/Bodzas/data_sample/"

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

def upload_and_process_features(features_file, data_file, data_folder, model_path, model_file, model_option):
    if features_file is not None:
        features = np.load(features_file)
    else:
        raise ValueError("Features file is required for this option.")
    
    if data_file is not None:
        data_folder = os.path.dirname(data_file.name)
    if not data_folder:
        data_folder = DEFAULT_DATA_PATH

    _, labels, class_names, image_paths = load_images(data_folder)
    umap_fig = create_interactive_umap_with_images(features, labels, image_paths, class_names)
    return umap_fig

def upload_and_process_data_and_model(features_file, data_file, data_folder, model_path, model_file, model_option):
    if model_file is not None:
        model_path = model_file.name
    if not model_path:
        model_path = DEFAULT_MODEL_PATH

    if data_file is not None:
        data_folder = os.path.dirname(data_file.name)
    if not data_folder:
        data_folder = DEFAULT_DATA_PATH
    
    modelname = model_options[model_option]
    model = get_dino_bloom(model_path, modelname).cuda()
    images, labels, class_names, image_paths = load_images(data_folder)
    images = images.cuda()
    
    with torch.no_grad():
        features = model(images).cpu().numpy()
    
    umap_fig = create_interactive_umap_with_images(features, labels, image_paths, class_names)
    return umap_fig

st.title("UMAP Visualization with DINOv2 Features")
option = st.radio("Choose an option", ["Upload features and data", "Upload data and model"])
model_option = st.selectbox("Select Model Type", ["DinoBloom S", "DinoBloom B", "DinoBloom L", "DinoBloom G"])

if option == "Upload features and data":
    features_file = st.file_uploader("Upload Features File (required)", type=["npy"])
    data_file = st.file_uploader("Upload Data Folder (optional)")
    data_folder = st.text_input("Data Folder Path (leave empty if uploading)", DEFAULT_DATA_PATH)
    model_path = None
    model_file = None
    if st.button("Visualize UMAP"):
        if features_file is not None:
            fig = upload_and_process_features(features_file, data_file, data_folder, model_path, model_file, model_option)
            st.plotly_chart(fig)
        else:
            st.error("Please upload a features file.")
else:
    features_file = None
    data_file = st.file_uploader("Upload Data Folder (optional)")
    data_folder = st.text_input("Data Folder Path (leave empty if uploading)", DEFAULT_DATA_PATH)
    model_file = st.file_uploader("Upload Model File (optional)", type=["pth"])
    model_path = st.text_input("Model Path (leave empty if uploading)", DEFAULT_MODEL_PATH)
    if st.button("Visualize UMAP"):
        if model_file is not None or model_path:
            fig = upload_and_process_data_and_model(features_file, data_file, data_folder, model_path, model_file, model_option)
            st.plotly_chart(fig)
        else:
            st.error("Please upload a model file or specify the model path.")
