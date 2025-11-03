# GNN-based Cancer Classification from Histopathology Images 

## Ting Jin

This project implements Graph Neural Networks (GNNs) for identifying biomakrers per cancer type from histopathology whole slide images (WSI). The pipeline processes tissue patches, converts them into graphs, and trains Graph Attention Networks (GAT) to predict cancer types and treatment responses.

## 🎯 Overview

This project addresses the challenge of cancer classification and treatment response prediction using histopathology images by:

1. **Extracting regions** from whole slide images (WSI)
2. **Converting tissue patches to graphs** using spatial KNN and feature similarity
3. **Training GNN models** for response status classification
4. **Analyzing and visualizing** model predictions and attention patterns

### Supported Cancer Types
- **Endometrial** cancer
- **Ovarian** cancer  
- **TNBC** (Triple-Negative Breast Cancer)
- **Cervical** cancer

## 📁 Project Structure

```
GNN/
├── step0_prepare_graphs/          # Graph construction from patches
│   ├── GNN_generataion.py       # Graph generation 
│
├── step1_GNN_training/   

```

## 🔧 Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies

Install the required packages:

```bash
pip install torch>=1.9.0
pip install torch-geometric>=2.0.0
pip install torch-scatter>=2.0.0
pip install torch-sparse>=0.6.0
pip install torch-cluster>=1.5.0
pip install torch-spline-conv>=2.0.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install scikit-learn>=1.0.0
pip install networkx>=2.6.0
pip install scikit-image
```

Or install from the requirements file:

```bash
pip install -r others/gcn_requirements.txt
```
## 🗂️ Data Format

### Input NPY Files

Each NPY file should contain a dictionary with:
```python
{
    'fseg_data': np.array,           # FSEG segmentation (values 0-10)
    'path_annotation': np.array,     # Binary mask of tissue regions
    'patches': list,                 # List of patch coordinates (minr, minc, maxr, maxc)
    'wsi_id': str,                   # Whole slide image ID
    'region_id': str                 # Region identifier
}
```

### Graph Data Format

Output graphs contain:
```python
{
    'node_features': np.array,        # Shape: (num_nodes, 11) - FSEG percentages
    'edges': list,                   # List of edge tuples (i, j)
    'patch_centers': np.array        # Shape: (num_nodes, 2) - Patch center coordinates
}

## 🔬 Research Applications

This project is designed for:
- **Cancer type classification** from histopathology images
- **Treatment response prediction** (responder vs non-responder)
- **Spatial pattern analysis** in tissue samples
- **Interpretable AI** for clinical decision support



For detailed documentation on specific components:
- **Graph-balanced splits**: See `pan_cancer/README_GRAPH_BALANCED.md`
- **Graph construction**: See `step0_prepare_graphs/fixed_coordinates.py` docstrings
- **Model architecture**: See `others/cancer_gnn_model.py`

