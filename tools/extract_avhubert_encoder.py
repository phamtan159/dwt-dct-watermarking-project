import torch
from pathlib import Path

def get_visual_encoder(checkpoint_path, device="cuda"):
    from avhubert.hubert import AVHubertModel
    from avhubert.hubert_cfg import AVHubertConfig

    state = torch.load(checkpoint_path, map_location="cpu")
    cfg = AVHubertConfig()

    model = AVHubertModel(cfg)
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    visual_encoder = model.encoder.video_frontend
    visual_encoder.eval()
    return visual_encoder.to(device)

if __name__ == "__main__":
    encoder = get_visual_encoder("pretrained/large_vox_iter5.pt", device="cpu")
    dummy = torch.randn(1, 1, 88, 88, 30)  
    with torch.no_grad():
        feat = encoder(dummy)   
    print("Feature shape:", feat.shape)
