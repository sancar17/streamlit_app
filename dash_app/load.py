import zipfile
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import os
import io
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import base64

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

class ZipDataset(Dataset):
    def __init__(self, zip_file):
        self.zip_file = zipfile.ZipFile(zip_file, 'r')
        self.image_files = [f for f in self.zip_file.namelist() if f.endswith('.bmp')]
        self.class_dirs = set(os.path.dirname(f) for f in self.image_files)
        self.class_names = sorted(self.class_dirs)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.preprocess_display = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        with self.zip_file.open(image_file) as file:
            img = Image.open(file).convert('RGB')
            class_name = os.path.dirname(image_file)
            label = self.class_to_idx[class_name]

            # Preprocess for model input
            img_tensor = self.preprocess(img)

            # Preprocess for display
            img_display = self.preprocess_display(img)
            img_bytes = io.BytesIO()
            transforms.ToPILImage()(img_display).save(img_bytes, format='PNG')
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        return img_tensor, label, img_base64, image_file

def load_images(zip_file):
    print("Loading images from ZIP file...")
    dataset = ZipDataset(zip_file)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    images = []
    labels = []
    image_paths = []
    valid_image_paths = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for batch in dataloader:
        img_tensors, batch_labels, img_base64s, file_paths = batch
        images.append(img_tensors.to(device))
        labels.extend(batch_labels.numpy())
        image_paths.extend(img_base64s)
        valid_image_paths.extend(file_paths)

    images = torch.cat(images)
    labels = np.array(labels)
    
    return images, labels, dataset.class_names, image_paths, valid_image_paths

def get_dino_bloom(modelpath, modelname="dinov2_vitb14"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = torch.load(modelpath, map_location=device)
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
    model = model.to(device)
    return model