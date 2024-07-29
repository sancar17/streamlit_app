import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import os
import numpy as np
import umap
import plotly.graph_objects as go
import gdown
import zipfile
import base64
from io import BytesIO
from streamlit.components.v1 import html

st.set_page_config(layout="wide")

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

# Check if a file exists
def check_if_file_exists(filepath):
    return os.path.isfile(filepath)

# Check if a directory exists
def check_if_directory_exists(dirpath):
    return os.path.isdir(dirpath)

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

    images_base64 = []
    for image_path in image_paths:
        image = Image.open(image_path)
        # Create small image for UMAP plot
        small_image = image.copy()
        small_image.thumbnail((50, 50))
        small_buffered = BytesIO()
        small_image.save(small_buffered, format="PNG")
        small_img_str = base64.b64encode(small_buffered.getvalue()).decode()
        
        # Create larger image for tooltip
        large_image = image.copy()
        large_image.thumbnail((200, 200))
        large_buffered = BytesIO()
        large_image.save(large_buffered, format="PNG")
        large_img_str = base64.b64encode(large_buffered.getvalue()).decode()
        
        images_base64.append({
            'small': f"data:image/png;base64,{small_img_str}",
            'large': f"data:image/png;base64,{large_img_str}"
        })

    fig = go.Figure()

    for img, (x, y), label in zip(images_base64, umap_data, labels):
        fig.add_layout_image(
            dict(
                source=img['small'],
                xref="x",
                yref="y",
                x=x,
                y=y,
                sizex=0.3,
                sizey=0.3,
                xanchor="center",
                yanchor="middle",
                layer="above"
            )
        )
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(size=10, opacity=0.5),
            hoverinfo='none',
            customdata=[label]
        ))

    fig.update_layout(
        title="UMAP Projection with Images",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white",
        showlegend=False,
        hovermode="closest"
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig, images_base64, class_names

# Your existing functions (get_dino_bloom, upload_and_process_data_and_model) remain unchanged

st.title("UMAP Visualization with DinoBloom Features")
option = st.radio("Choose an option", ["Use Features (Not Implemented)", "Use Model"])

if option == "Use Features":
    st.write("Feature-based visualization is not implemented yet.")
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
            fig, images_base64, class_names = upload_and_process_data_and_model(model_source, model_file, data_source, data_file)
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add custom tooltip
            tooltip_html = f"""
            <div id="tooltip" style="display: none; position: absolute; background: white; border: 1px solid black; padding: 5px; z-index: 1000;"></div>
            <script>
            const tooltip = document.getElementById('tooltip');
            const plotlyElement = document.querySelector('.js-plotly-plot');
            const images = {images_base64};
            const classNames = {class_names};

            plotlyElement.on('plotly_hover', function(data) {{
                const point = data.points[0];
                const idx = point.pointNumber;
                
                const xPos = point.xaxis.d2p(point.x) + point.xaxis._offset;
                const yPos = point.yaxis.d2p(point.y) + point.yaxis._offset;
                
                tooltip.innerHTML = `
                    <img src="${{images[idx].large}}" style="width:200px">
                    <p>${{classNames[point.customdata[0]]}}</p>
                `;
                tooltip.style.left = (xPos + 10) + 'px';
                tooltip.style.top = (yPos + 10) + 'px';
                tooltip.style.display = 'block';
            }});

            plotlyElement.on('plotly_unhover', function(data) {{
                tooltip.style.display = 'none';
            }});
            </script>
            """
            
            html(tooltip_html, height=0)
        else:
            st.error("Please select a model or upload a model file.")