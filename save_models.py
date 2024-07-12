import torch
import torch.nn as nn
import os

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

def save_final_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def main():
    models_path = "./models"
    model_files = {
        "DinoBloom S": "DinoBloom-S.pth",
        "DinoBloom B": "DinoBloom-B.pth",
        "DinoBloom L": "DinoBloom-L.pth",
        "DinoBloom G": "DinoBloom-G.pth"
    }

    for model_key, model_file in model_files.items():
        model_path = os.path.join(models_path, model_file)
        if os.path.exists(model_path):
            model_name = model_options[model_key]
            print(f"Processing {model_file} with model name {model_name}")
            model = get_dino_bloom(model_path, model_name)
            save_path = os.path.join(models_path, f"final_{model_file}")
            save_final_model(model, save_path)
            print(f"Saved final model to {save_path}")
        else:
            print(f"Model file {model_file} does not exist in the path {models_path}")

if __name__ == "__main__":
    main()
