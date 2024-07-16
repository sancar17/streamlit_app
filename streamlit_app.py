import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from tqdm import tqdm
import gdown
import zipfile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Google Drive URLs 
GDRIVE_URLS = {
    "sample_data": "https://drive.google.com/uc?id=1c-OBD9x_RT_VX0GZUbmOeEIFgpEdNNRH",
    "DinoBloom S": "https://drive.google.com/uc?id=1gedjQGhf4FiYpF1tP40ugMaYc0t6GhZZ",
    "DinoBloom B": "https://drive.google.com/uc?id=1vs1DDpl3O93C_AwLLjaYSiKAI-N_Uitc",
    "DinoBloom L": "https://drive.google.com/uc?id=1eXGCZzDez85ip4LEX1VIHe4TBmpuXaHY",
    "DinoBloom G": "https://drive.google.com/uc?id=1-C-ip2qrKsp4eYBebw3ItWuu63crUitE"
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

# Function to download file from Google Drive
def download_from_gdrive(gdrive_url, download_path):
    gdown.download(gdrive_url, download_path, quiet=False)

def load_images(data_folder):
    st.write(f"Loading images from {data_folder}")
    image_paths = []
    labels = []
    # Adjusted to look for class folders under the nested data_samples folder
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

def create_umap_with_images(data, labels, image_paths, class_names):
    reducer = umap.UMAP()
    umap_data = reducer.fit_transform(data)

    plt.figure(figsize=(20, 14))
    ax = plt.subplot(111)
    
    # Plot the images at their corresponding UMAP coordinates
    for i in range(len(umap_data)):
        image = Image.open(image_paths[i])
        image = image.resize((40, 40), Image.LANCZOS)  # Increase the size of the images
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(image, cmap="gray"),
            umap_data[i],
            frameon=False
        )
        ax.add_artist(imagebox)
    
    plt.title("UMAP Projection with Images")
    
    # Annotate clusters with labels
    for label in np.unique(labels):
        label_mask = (labels == label)
        label_data = umap_data[label_mask]
        x_mean, y_mean = np.mean(label_data, axis=0)
        ax.text(x_mean, y_mean, class_names[label], fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlim(umap_data[:, 0].min() - 1, umap_data[:, 0].max() + 1)
    ax.set_ylim(umap_data[:, 1].min() - 1, umap_data[:, 1].max() + 1)

    plt.savefig("umap_visualization_with_images.png", bbox_inches='tight')
    return "umap_visualization_with_images.png"

def upload_and_process_features(features_file, data_source, data_file):
    if features_file is not None:
        features = np.load(features_file)
    else:
        raise ValueError("Features file is required for this option.")
    
    if data_source == "Sample Data":
        data_path = "sample_data"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        download_path = os.path.join(data_path, "sample_data.zip")
        download_from_gdrive(GDRIVE_URLS["sample_data"], download_path)
        # Unzip the downloaded file
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    elif data_file is not None:
        data_path = os.path.dirname(data_file.name)
    else:
        raise ValueError("Data source is required for this option.")

    _, labels, class_names, image_paths = load_images(data_path)
    umap_image_path = create_umap_with_images(features, labels, image_paths, class_names)
    return umap_image_path

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
    if model_source != "Upload Model":
        model_key = model_source
        st.write("Downloading model:", GDRIVE_URLS[model_key])
        model_path = f"{model_key.replace(' ', '_')}.pth"
        download_from_gdrive(GDRIVE_URLS[model_key], model_path)
        model = get_dino_bloom(model_path, model_options[model_key])
    elif model_file is not None:
        model_path = model_file.name
        model = get_dino_bloom(model_path, model_options[model_source])
    else:
        raise ValueError("Model source is required for this option.")

    if data_source == "Sample Data":
        data_path = "sample_data"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        download_path = os.path.join(data_path, "sample_data.zip")
        st.write("Downloading data:", GDRIVE_URLS["sample_data"])
        download_from_gdrive(GDRIVE_URLS["sample_data"], download_path)
        # Unzip the downloaded file
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    elif data_file is not None:
        data_path = os.path.dirname(data_file.name)
    else:
        raise ValueError("Data source is required for this option.")
    
    images, labels, class_names, image_paths = load_images(data_path)
    images = images
    
    with torch.no_grad():
        features = model(images).cpu().numpy()
    
    umap_image_path = create_umap_with_images(features, labels, image_paths, class_names)
    return umap_image_path

st.title("UMAP Visualization with DinoBloom Features")
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
            umap_image_path = upload_and_process_features(features_file, data_source, data_file)
            st.image(umap_image_path)
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
            umap_image_path = upload_and_process_data_and_model(model_source, model_file, data_source, data_file)
            st.image(umap_image_path)
        else:
            st.error("Please select a model or upload a model file.")
