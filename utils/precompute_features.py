"""
Precompute features for gallery images and build a FAISS index.
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure project root on sys.path (works with Windows Python called from bash)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# torch with helpful error if missing
try:
    import torch
except Exception as e:
    print("ERROR: PyTorch is not importable in this interpreter.")
    print("Python exe:", sys.executable)
    print("Exception:", repr(e))
    sys.exit(1)

from src.model import VGGFineTuned
from utils.image_utils import preprocess_image, extract_features
from utils.faiss_utils import build_faiss_index, save_faiss_index


# ---------- DATA ROOT AUTODETECT ----------
def _detect_data_root() -> Path:
    """
    Find the directory that actually contains '101_ObjectCategories'.
    Returns the parent dir of 101_ObjectCategories.
    """
    # Common locations
    cands = [
        PROJECT_ROOT / "caltech101",
        PROJECT_ROOT,  # maybe directly here
    ]
    for c in cands:
        if (c / "101_ObjectCategories").exists():
            return c

    # Search anywhere under the project for '101_ObjectCategories'
    hit = next(PROJECT_ROOT.rglob("101_ObjectCategories"), None)
    if hit:
        return hit.parent

    # Fallback to project root (will cause warnings if not present)
    return PROJECT_ROOT


DATA_ROOT = _detect_data_root()
print(f"Detected DATA_ROOT: {DATA_ROOT.as_posix()}")


def resolve_img_path(img_path_str: str) -> Path | None:
    """
    Map JSON path (often 'caltech101/101_ObjectCategories/...') to the real filesystem.
    """
    p = Path(img_path_str.replace("\\", "/"))

    # If absolute and exists, just use it
    if p.is_absolute() and p.exists():
        return p

    parts = p.parts

    # Normalize: strip an optional leading 'caltech101'
    if parts and parts[0] == "caltech101":
        p = Path(*parts[1:])
        parts = p.parts

    # If path starts with '101_ObjectCategories/...', anchor at DATA_ROOT
    if parts and parts[0] == "101_ObjectCategories":
        cand = DATA_ROOT / p
        if cand.exists():
            return cand

    # Try as-is under PROJECT_ROOT
    cand = PROJECT_ROOT / p
    if cand.exists():
        return cand

    # Try under DATA_ROOT as a last resort
    cand = DATA_ROOT / p
    if cand.exists():
        return cand

    return None
# -----------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Precompute features for gallery images")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--data", type=str, required=True, help="Path to the gallery data JSON file")
    parser.add_argument("--faiss", type=str, required=True, help="Path to save the FAISS index")
    parser.add_argument("--pickle", type=str, required=True, help="Path to save features as a pickle dictionary")
    parser.add_argument("--num-per-class", type=int, default=10, help="Images per class to index")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    data_path = PROJECT_ROOT / args.data
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    # Ensure output dirs
    (PROJECT_ROOT / Path(args.faiss)).parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / Path(args.pickle)).parent.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Interpreter:", sys.executable)
    print(f"Using device: {device}")

    # Load model
    model = VGGFineTuned(num_classes=102, embedding_size=128, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Load data JSON
    with open(data_path, "r") as f:
        gallery_data = json.load(f)

    if "train" not in gallery_data or "categories" not in gallery_data:
        print("Error: The data file must have 'train' and 'categories' keys")
        return

    categories = gallery_data["categories"]
    print(f"Found {len(categories)} categories")

    # Group by label
    images_by_category: dict[int, list[tuple[int, str]]] = {}
    for i, (label, img_path) in enumerate(gallery_data["train"]):
        images_by_category.setdefault(label, []).append((i, img_path))

    # Sample per class
    selected = []
    random.seed(42)
    for label, imgs in images_by_category.items():
        k = min(args.num_per_class, len(imgs))
        sampled = random.sample(imgs, k) if len(imgs) > k else imgs
        selected.extend(sampled)
        name = categories[label] if label < len(categories) else f"Category {label}"
        print(f"Category {name}: Selected {k} of {len(imgs)}")

    print(f"\nProcessing {len(selected)} images for FAISS index...")

    all_features = []
    all_paths = []
    features_dict = {}

    for _, img_rel in tqdm(selected):
        try:
            full = resolve_img_path(img_rel)
            if full is None:
                print(f"Warning: Could not find image at {img_rel}")
                continue

            image = Image.open(full).convert("RGB")
            image_tensor = preprocess_image(image, device)
            feats = extract_features(model, image_tensor, device)  # (1, D) numpy

            all_features.append(feats.astype("float32"))
            all_paths.append(str(full))
            features_dict[str(full)] = feats[0].astype("float32")
        except Exception as e:
            print(f"Error processing {img_rel}: {e}")

    if not all_features:
        print("No features extracted. Check that the image paths in the data file are correct.")
        return

    all_features = np.vstack(all_features)  # (N, D) float32
    print(f"Extracted features shape: {all_features.shape}")

    # Save paths + metadata
    paths_file = (PROJECT_ROOT / Path(args.faiss)).parent / "features_paths.json"
    paths_with_metadata = []

    # Robust label mapping (absolute vs relative)
    for p in all_paths:
        p_posix = Path(p).resolve().as_posix()
        label = None
        for train_label, train_path in gallery_data["train"]:
            tp = Path(str(train_path).replace("\\", "/")).as_posix()
            # allow leading 'caltech101/' in JSON
            if tp.startswith("caltech101/"):
                tp = tp[len("caltech101/"):]
            if p_posix.endswith(tp):
                label = train_label
                break

        cat_name = categories[label] if (label is not None and label < len(categories)) else "Unknown"
        paths_with_metadata.append({"path": p, "label": label, "category": cat_name})

    with open(paths_file, "w") as f:
        json.dump(paths_with_metadata, f)
    print(f"Image paths with metadata saved to {paths_file} (contains {len(paths_with_metadata)} entries)")

    # Build + save FAISS
    index = build_faiss_index(all_features)
    save_faiss_index(index, str(PROJECT_ROOT / args.faiss))
    print(f"FAISS index saved to {args.faiss}")

    # Optional pickle
    if args.pickle:
        with open(PROJECT_ROOT / args.pickle, "wb") as f:
            pickle.dump(features_dict, f)
        print(f"Features dictionary saved to {args.pickle} (contains {len(features_dict)} entries)")


if __name__ == "__main__":
    main()
