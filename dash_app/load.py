import zipfile
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import os
import io
import torch.nn as nn

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

def load_images(zip_file):
    print("Loading images from ZIP file...")

    with zipfile.ZipFile(zip_file, 'r') as z:
        # List of files in the ZIP archive
        file_names = z.namelist()
        
        # Filter to only include image files and directories
        image_files = [f for f in file_names if f.endswith('.bmp')]
        class_dirs = set(os.path.dirname(f) for f in image_files)

        # Extract class names from directories
        class_names = sorted(class_dirs)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        # Initialize lists for storing images, labels, and paths
        images = []
        labels = []
        valid_image_paths = []

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for image_file in image_files:
            try:
                with z.open(image_file) as file:
                    img = Image.open(file).convert('RGB')
                    class_name = os.path.dirname(image_file)
                    label = class_to_idx[class_name]
                    images.append(preprocess(img))
                    labels.append(label)
                    valid_image_paths.append(image_file)
            except (OSError, IOError) as e:
                print(f"Skipping corrupted image: {image_file}, error: {e}")
        
        images = torch.stack(images)
        labels = np.array(labels)
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