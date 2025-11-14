import faiss
import numpy as np

def build_faiss_index(features: np.ndarray):
    """Build a simple L2 FAISS index from a (N, D) feature matrix."""
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features.astype('float32'))
    return index

def save_faiss_index(index, path):
    """Save FAISS index to disk."""
    faiss.write_index(index, path)

def load_faiss_index(path):
    """Load FAISS index from disk."""
    return faiss.read_index(path)

def search_similar_images(query_vector, index, k=5):
    """Return distances and indices of the top-k nearest images."""
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances[0], indices[0]
