#!/bin/bash

# Create necessary directories
mkdir -p data
mkdir -p weights

# Default paths
MODEL_PATH="weights/model.pth"
DATA_FILE="train_val.json"
FAISS_PATH="data/faiss_index.bin"
PICKLE_PATH="data/features.pickle"
NUM_PER_CLASS=20

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-per-class)
      NUM_PER_CLASS="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

echo "Image Retrieval: Precomputing Features"
echo "====================================="
echo "This script will extract features from your training images and"
echo "create a FAISS index for fast similarity search."
echo "For efficiency, only ${NUM_PER_CLASS} random images per class will be used."
echo

# Checks
if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  exit 1
fi
if [ ! -f "$DATA_FILE" ]; then
  echo "ERROR: Data file not found at $DATA_FILE"
  exit 1
fi

echo "Starting feature extraction (this may take a while)..."
echo "Using model: $MODEL_PATH"
echo "Using data: $DATA_FILE"
echo "Creating FAISS index at: $FAISS_PATH"
echo "Saving features dictionary at: $PICKLE_PATH"
echo "Using GPU if available (auto-detected)"
echo

# Always use Windows Python (torch installed there)
PYWIN="/mnt/c/Users/victo/AppData/Local/Programs/Python/Python313/python.exe"

# Print interpreter sanity info
"$PYWIN" - <<'PY'
import sys, torch
print("Windows Python:", sys.executable)
print("Torch version:", getattr(torch, "__version__", "not found"))
print("CUDA available:", torch.cuda.is_available())
PY

# Run
"$PYWIN" utils/precompute_features.py \
  --model "$MODEL_PATH" \
  --data "$DATA_FILE" \
  --faiss "$FAISS_PATH" \
  --pickle "$PICKLE_PATH" \
  --num-per-class "$NUM_PER_CLASS"

if [ $? -eq 0 ]; then
  echo
  echo "Feature extraction completed successfully!"
else
  echo
  echo "Feature extraction failed. Please check the error messages above."
fi
