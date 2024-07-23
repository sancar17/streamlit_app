import torch

# Function to save model to local path
def save_model(model_name, save_path):
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    torch.save(model, save_path)
    print(f"Model {model_name} saved to {save_path}")

# Paths to save the models
model_paths = {
    "dinov2_vits14": "./dinov2_vits14.pt",
    "dinov2_vitb14": "./dinov2_vitb14.pt",
    "dinov2_vitl14": "./dinov2_vitl14.pt",
    "dinov2_vitg14": "./dinov2_vitg14.pt"
}

# Download and save all models
for model_name, save_path in model_paths.items():
    save_model(model_name, save_path)

print("All models have been downloaded and saved.")
