import torch

# Function to save the model state dict to local path
def save_model_state_dict(model_name, save_path):
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model state dict {model_name} saved to {save_path}")

# Paths to save the model state dicts
model_paths = {
    "dinov2_vits14": "dinov2_vits14_state_dict.pth",
    "dinov2_vitb14": "dinov2_vitb14_state_dict.pth",
    "dinov2_vitl14": "dinov2_vitl14_state_dict.pth",
    "dinov2_vitg14": "dinov2_vitg14_state_dict.pth"
}

# Download and save all model state dicts
for model_name, save_path in model_paths.items():
    save_model_state_dict(model_name, save_path)

print("All model state dicts have been downloaded and saved.")
