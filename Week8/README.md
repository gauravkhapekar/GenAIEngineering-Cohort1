# Week 8: Embeddings and Fusion in Multimodal AI

This week focuses on understanding and implementing embeddings for different modalities (text, images, and audio) and exploring fusion techniques to combine multimodal information effectively.

## Learning Objectives

- Understand how to create embeddings for text, images, and audio using pre-trained models
- Learn different fusion strategies for combining multimodal embeddings
- Implement late fusion techniques for multimodal classification
- Explore visualization techniques for high-dimensional embeddings
- Build complete multimodal pipelines using Hugging Face models

## Repository Setup

### Clone the GitHub Repository

If you haven't already, clone the repository using Git:

```bash
# Clone the repository
git clone https://github.com/outskill-git/GenAIEngineering-Cohort1

# Navigate into the repository folder
cd GenAIEngineering-Cohort1/Week8
```

### Create a Virtual Environment

Create and activate a Python virtual environment:

#### For Windows:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

#### For macOS/Linux:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Install Requirements

Install the packages listed in requirements.txt:

```bash
# Install required packages
pip install -r requirements.txt
```

### Verify Installation

Verify that everything is set up correctly:

```bash
# Check installed packages
pip list

# Test key imports
python -c "import torch; import transformers; import datasets; print('Setup successful!')"
```

## Session Content

### Session 1 Notebooks:

1. **text_embeddings.ipynb** [Google Colab](https://colab.research.google.com/drive/1mClrNFwUeztQjUL4NXZslr9nEoeD67sj?usp=sharing)
   - Creating dense vector embeddings for text using pre-trained transformer models
   - Working with sentence transformers and BERT-based models
   - Understanding tokenization, pooling strategies, and embedding normalization
   - Computing text similarity using cosine similarity
   - Comparing different text embedding models

2. **image_embedding.ipynb** [Google Colab](https://colab.research.google.com/drive/1AQPc0q6kN_ADS_Fbu-FIE_YrnQXb-G0o?usp=sharing)
   - Generating image embeddings using Vision Transformers (ViT) and other models
   - Loading and preprocessing images from Hugging Face datasets (MNIST)
   - Understanding image preprocessing pipelines and tensor operations
   - Visualizing high-dimensional embeddings using PCA and t-SNE
   - Computing image similarity matrices and clustering analysis

3. **audio_embedding.ipynb** [Google Colab](https://colab.research.google.com/drive/1ExzturxhZDXktxqSiMWxv0cSvwQ8-mVt?usp=sharing)
   - Creating audio embeddings using Wav2Vec2 and other audio models
   - Working with speech commands dataset and audio preprocessing
   - Understanding audio feature extraction and sampling rates
   - Visualizing audio embeddings and analyzing acoustic similarities
   - Handling different audio formats and resampling techniques

4. **late_fusion.ipynb** [Google Colab](https://colab.research.google.com/drive/1F9Tek26MLHys1uE5s9YFTzDqcjgwVA63?usp=sharing)
   - Implementing late fusion strategies for multimodal classification
   - Combining text and image embeddings from paired datasets (Flickr8k)
   - Building separate classifiers for each modality
   - Weighted fusion techniques and threshold-based binary classification
   - Detailed analysis of fusion calculations and decision boundaries

## Key Concepts Covered

### Embeddings
- **Text Embeddings**: Sentence transformers, BERT, tokenization, mean pooling
- **Image Embeddings**: Vision Transformers, CNN features, image preprocessing
- **Audio Embeddings**: Wav2Vec2, spectral features, audio preprocessing
- **Embedding Properties**: Dimensionality, normalization, similarity metrics

### Fusion Techniques
- **Late Fusion**: Combining predictions from separate modality-specific models
- **Weighted Fusion**: Assigning different importance to different modalities
- **Score Aggregation**: Methods for combining confidence scores
- **Threshold-based Classification**: Converting continuous scores to discrete labels

### Visualization and Analysis
- **Dimensionality Reduction**: PCA, t-SNE for embedding visualization
- **Similarity Analysis**: Cosine similarity, distance metrics
- **Clustering**: Understanding embedding space structure
- **Performance Evaluation**: Classification metrics and fusion effectiveness
  
### Getting Help:

- Check that your virtual environment is activated (you should see `(venv)` in your terminal prompt)
- Verify all packages are installed with `pip list`
- Restart Jupyter kernel if you encounter runtime errors
- Check model compatibility with your PyTorch version
- Ensure sufficient disk space for model downloads

## Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

