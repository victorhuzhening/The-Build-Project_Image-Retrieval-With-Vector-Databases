import torch
from torchvision import transforms

# Same normalization used for ResNet/VGG
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image, device):
    """Convert a PIL image to a normalized tensor and add batch dimension."""
    tensor = _preprocess(image).unsqueeze(0).to(device)
    return tensor

@torch.no_grad()
def extract_features(model, image_tensor, device):
    """Return normalized embedding from model.extract_features()."""
    model.eval()
    feats = model.extract_features(image_tensor.to(device))
    return feats.cpu().numpy()