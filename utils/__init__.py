# utils/__init__.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

# ---- FAISS (CPU) ----
try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    faiss = None
    _faiss_import_error = e
else:
    _faiss_import_error = None


# =========================
# Image preprocessing / features
# =========================

# Use the same normalization as VGG/ResNet
_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Convert a PIL image to a normalized tensor (1, 3, 224, 224) on device.
    """
    return _PREPROCESS(image).unsqueeze(0).to(device)

@torch.no_grad()
def extract_features(model, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Get a (1, D) numpy embedding from your model. Assumes model has .extract_features().
    If your model returns unnormalized vectors, you can L2-normalize here.
    """
    model.eval()
    feats = model.extract_features(image_tensor.to(device))  # (1, D) torch
    # Ensure CPU float32 numpy
    feats = feats.detach().cpu().to(torch.float32).numpy()
    # If you want to be extra safe about cosine similarity:
    # feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return feats


# =========================
# FAISS helpers
# =========================

def _ensure_faiss():
    if faiss is None:
        raise ImportError(
            "faiss is not available. Install with: pip install faiss-cpu\n"
            f"Original error: {_faiss_import_error}"
        )

def build_faiss_index(features: np.ndarray, metric: str = "ip"):
    """
    Build a FAISS index from (N, D) float32 features.
    metric='ip' (inner product) assumes features are L2-normalized -> cosine similarity.
    metric='l2' for L2 distance.
    """
    _ensure_faiss()
    assert features.dtype == np.float32, "features must be float32"
    d = features.shape[1]
    if metric.lower() == "ip":
        index = faiss.IndexFlatIP(d)
    elif metric.lower() == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("metric must be 'ip' or 'l2'")
    index.add(features)
    return index

def save_faiss_index(index, path: str):
    _ensure_faiss()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_faiss_index(path: str):
    """
    Load a FAISS index from disk. Returns None if not found or FAISS missing.
    """
    if not os.path.exists(path):
        return None
    _ensure_faiss()
    return faiss.read_index(path)

def search_similar_images(query_vector: np.ndarray, index, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    query_vector: (1, D) float32
    Returns: (similarities_or_distances[ k ], indices[ k ])
    """
    if query_vector.dtype != np.float32:
        query_vector = query_vector.astype(np.float32)
    sims, idxs = index.search(query_vector, k)
    return sims[0], idxs[0]


# =========================
# Streamlit display helper
# =========================

def display_results(similarities: np.ndarray,
                    indices: np.ndarray,
                    indexed_paths: List[dict],
                    grid_cols: int = 5):
    """
    Show a grid of search results in Streamlit.
    - similarities: array of scores (cosine if IP, negative L2 if L2)
    - indices: FAISS indices into indexed_paths
    - indexed_paths: list of dicts like {"path": "...", "label": int, "category": "name"}
    """
    try:
        import streamlit as st
    except ImportError:
        raise RuntimeError("display_results requires streamlit (pip install streamlit)")

    # Build rows of columns
    n = len(indices)
    for start in range(0, n, grid_cols):
        cols = st.columns(min(grid_cols, n - start))
        for j, col in enumerate(cols, start=start):
            idx = indices[j]
            if idx < 0 or idx >= len(indexed_paths):
                col.write("Index out of bounds")
                continue

            meta = indexed_paths[idx]
            path = meta.get("path", "")
            category = meta.get("category", "Unknown")
            score = similarities[j]

            # Load image safely
            try:
                img = Image.open(path).convert("RGB")
                col.image(img, use_column_width=True)
            except Exception:
                col.write(f"Missing image:\n{path}")

            # Caption with category + score
            col.caption(f"{category}\nScore: {score:.4f}\n{Path(path).name}")
