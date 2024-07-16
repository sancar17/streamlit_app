import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import os
import numpy as np
import umap
import plotly.express as px
from tqdm import tqdm
import requests
from io import BytesIO
import zipfile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

st.title("UMAP Visualization with DINOv2 Features")
option = st.radio("Choose an option", ["Use Features", "Use Model"])

if option == "Use Features":
    features_file = st.file_uploader("Upload Features File (required)", type=["npy"])
    data_source = st.selectbox("Choose Data Source", ["Sample Data", "Upload Data"])
    if data_source == "Upload Data":
        data_file = st.file_uploader("Upload Data Folder (optional)")
    else:
        data_file = None
    '''if st.button("Visualize UMAP"):
        if features_file is not None:
            fig = upload_and_process_features(features_file, data_source, data_file)
            st.plotly_chart(fig)
        else:
            st.error("Please upload a features file.")'''
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
    '''if st.button("Visualize UMAP"):
        if model_source != "Upload Model" or model_file is not None:
            fig = upload_and_process_data_and_model(model_source, model_file, data_source, data_file)
            st.plotly_chart(fig)
        else:
            st.error("Please select a model or upload a model file.")'''
