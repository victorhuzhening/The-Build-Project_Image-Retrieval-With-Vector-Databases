import streamlit as st
import torch
import os
import json
import numpy as np
from PIL import Image

# Import our modules
from src.model import VGGFineTuned
from utils import (
    preprocess_image, extract_features,
    build_faiss_index, load_faiss_index, save_faiss_index, search_similar_images,
    display_results
)

# Set page configuration
st.set_page_config(
    page_title="Image Retrieval Demo", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths (default)
MODEL_DIR = "weights"
DATA_DIR = "data"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
DEFAULT_FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DEFAULT_FEATURES_PATHS_FILE = os.path.join(DATA_DIR, "features_paths.json")

@st.cache_resource
def load_model(model_path, num_classes=102):
    """Load the trained model"""
    # Determine device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"Using device: {device}")
    
    model = VGGFineTuned(num_classes=num_classes, embedding_size=128, pretrained=False).to(device)
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("Model loaded successfully!")
        else:
            st.error(f"Model not found at {model_path}. Please run precompute.sh first.")
            return None, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device
    
    model.eval()
    return model, device

def main():
    # Set up header
    st.markdown("<h1 style='text-align: center;'>AI Powered Image Retrieval Demo</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Find Similar Images using Vector Databases</p>", unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # Custom path inputs
    st.sidebar.subheader("Custom Paths")
    faiss_index_path = st.sidebar.text_input("FAISS Index Path", value=DEFAULT_FAISS_INDEX_PATH)
    features_paths_file = st.sidebar.text_input("Features Paths File", value=DEFAULT_FEATURES_PATHS_FILE)
    
    # Settings
    num_results = st.sidebar.slider("Number of Results", min_value=1, max_value=10, value=5)
    show_all_categories = st.sidebar.checkbox("Show All Categories", value=False)
    
    # Upload image section
    st.header("Upload an Image")
    
    # Create columns for upload and buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
    with col2:
        search_button = st.button("Search")
        
    with col3:
        clear_button = st.button("Clear")
    
    # Initialize session state for results
    if 'results_displayed' not in st.session_state:
        st.session_state.results_displayed = False
        
    # Display all categories if requested
    if show_all_categories and os.path.exists(features_paths_file):
        try:
            with open(features_paths_file, 'r') as f:
                indexed_paths = json.load(f)
                
            # Extract all unique categories
            categories = set()
            for item in indexed_paths:
                if "category" in item:
                    categories.add(item["category"])
            
            # Display categories
            st.sidebar.subheader("Available Categories")
            
            # Create a grid display for categories
            category_list = sorted(list(categories))
            rows = [category_list[i:i+2] for i in range(0, len(category_list), 2)]
            
            for row in rows:
                cols = st.sidebar.columns(2)
                for i, category in enumerate(row):
                    cols[i].write(f"• {category}")
        except Exception as e:
            st.sidebar.warning(f"Could not load categories: {e}")
    
    # Clear button logic
    if clear_button:
        # Reset all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Only load model and data if needed
    if uploaded_file is not None or search_button:
        # Load model and required data
        model, device = load_model(MODEL_PATH)
        if not model:
            st.error("Model not found. Please run precompute.sh first.")
            st.stop()
        
        # Try to load FAISS index
        faiss_index = load_faiss_index(faiss_index_path)
        if faiss_index is None:
            st.error(f"FAISS index not found at {faiss_index_path}. Please run precompute.sh first.")
            st.stop()
            
        # Try to load paths with metadata
        if os.path.exists(features_paths_file):
            try:
                with open(features_paths_file, 'r') as f:
                    indexed_paths = json.load(f)
                st.sidebar.info(f"FAISS index contains {len(indexed_paths)} images")
            except Exception as e:
                st.warning(f"Error loading features paths: {e}")
                indexed_paths = None
        else:
            st.warning(f"Features paths file not found at {features_paths_file}. Results may not be accurate.")
            indexed_paths = None
        
        # Check if we have paths data
        if not indexed_paths:
            st.error("Required files not found. Please run precompute.sh first.")
            st.stop()
        
        # Only perform search when button is clicked AND there's an uploaded file
        if search_button and uploaded_file:
            # Process image and display results
            query_image = Image.open(uploaded_file).convert('RGB')
            
            # Display the uploaded image in the sidebar
            st.sidebar.header("Uploaded Image")
            st.sidebar.image(query_image, use_column_width=True)
            
            # Process the query image
            with st.spinner("Processing image..."):
                # Process the query image with the model
                query_tensor = preprocess_image(query_image, device)
                query_feature = extract_features(model, query_tensor, device)
                
                # Search for similar images
                similarities, indices = search_similar_images(query_feature, faiss_index, k=num_results)
                
                # Display results
                display_results(similarities, indices, indexed_paths)
                
                # Mark that we've displayed results
                st.session_state.results_displayed = True
        elif search_button and not uploaded_file:
            st.warning("Please upload an image first before searching.")
    
    # Footer
    st.markdown("---")
    

if __name__ == "__main__":
    main()