## Build Project: Image Retrieval With Vector Databases ##

An image retrieval system using FAISS fast vector indexing for fast look-up, a front-end built using Streamlit Python library, and a fine-tuned VGG-16 model as the backbone for learning image embeddings. This repo showcases my work with The Build Fellowship under the supervision of Kamalesh Kalirathinam.


### Overview

#### Model

The backbone of this project is the VGG-16 model fine-tuned at 10 epochs with all layers weights frozen except for the last few. A model training script and model definition is included in this repo, which you may tweak for desired results. VGG-16 is chosen specifically for its balance between learning ability and computational efficiency as the author of this repo trained locally on a single NVIDIA RTX 4060 GPU.

#### Image Retrieval

The fine-tuned model generates a 128-dimensional embedding for each image, which is used to both extract information from the input image and perform similarity search between input and processed images. 

#### FAISS Vector Search

Suppors precomputing FAISS indexes for processed image embeddings to enable fast similarity search between input image embedding and desired image embeddings. Image embeddings from the desired dataset is precomputed and stored locally for fast look-up.

#### Dataset

All input images are compared against the Caltech101 [1] dataset, which contains thousands of images across 101 object categories organized in image folder format, where the folder name is the label for supervised learning. This dataset can be accessed via PyTorch API or downloaded manually from Kaggle: https://www.kaggle.com/datasets/imbikramsaha/caltech-101.

To ensure smooth execution, your dataset folder should look like this:

caltech101
 - 101_ObjectCategories
	-accordion
	-airplanes
	-anchor
	...
	

### How to run locally

Train the model. You will need to manually replace the dataset directory in the script:

\begin{lstlisting}[language=bash]
python train_model.py
\end{lstlisting}

Run FAISS index building script:

\begin{lstlisting[language=bash]
./precompute.sh
\end{lstlisting}

Build and deploy webapp:

\begin{lstlisting[language=bash]
python app.py
\end{lstlisting}


### References

[1] Li, F.-F., Andreeto, M., Ranzato, M., & Perona, P. (2022). Caltech 101 (1.0) [Data set]. CaltechDATA. https://doi.org/10.22002/D1.20086
