# Uploading Agricultural AI Models to Hugging Face

## Overview
This guide walks you through uploading your trained graph embedding models (GCN, TransE, DistMult, etc.) to Hugging Face Hub for versioning and sharing.

## Prerequisites

1. **Hugging Face Account**: Create one at https://huggingface.co/join
2. **Python 3.7+** installed
3. **Hugging Face CLI** or `huggingface_hub` library

---

## Step 1: Install Required Packages

Open your terminal and install the Hugging Face Hub library:

```bash
pip install huggingface_hub
```

Or if using a requirements file:

```bash
pip install huggingface_hub --upgrade
```

---

## Step 2: Login to Hugging Face

### Option A: Using Command Line (Recommended)

```bash
huggingface-cli login
```

You'll be prompted to:
1. Enter your Hugging Face token (get it from https://huggingface.co/settings/tokens)
2. The token will be saved for future use

### Option B: Using Python Script

```python
from huggingface_hub import login

login()
# Enter your token when prompted
```

---

## Step 3: Create a Repository on Hugging Face

### Via Web Interface (Easiest)
1. Go to https://huggingface.co/new
2. Fill in:
   - **Repository name**: `agricultural-ai-graph-models` (or your preferred name)
   - **Repository type**: Select "Model"
   - **Visibility**: Public or Private (your choice)
3. Click "Create repository"

### Via Command Line
```bash
huggingface-cli repo create agricultural-ai-graph-models --type model
```

---

## Step 4: Prepare Model Files

Your models are located in:
- `deployment/models/` or
- `deployment/processed/trained_models/`

**Files to upload**:
1. `best_model.pth` - Best performing model (GCN)
2. `gcn_model.pth` - Graph Convolutional Network
3. `transe_model.pth` - TransE model
4. `distmult_model.pth` - DistMult model
5. `complex_model.pth` - ComplEx model
6. `graphsage_model.pth` - GraphSAGE model
7. `model_metadata.json` - Entity/relation mappings
8. `best_model_info.json` - Buil model information

---

## Step 5: Create Upload Script

Create a new Python file: `upload_to_huggingface.py`

```python
#!/usr/bin/env python3
"""
Upload Agricultural AI Graph Models to Hugging Face Hub
"""

from huggingface_hub import HfApi, login
import os

# Login first (if not already logged in)
try:
    login()
    print("✓ Successfully logged in to Hugging Face")
except Exception as e:
    print(f"Login error: {e}")
    print("Please run: huggingface-cli login")

# Initialize API
api = HfApi()

# Repository name (change this to your repository name)
repo_id = "YOUR_USERNAME/agricultural-ai-graph-models"

# Model files directory
models_dir = "deployment/models"  # or "deployment deception/processed/trained_models"

print(f"\n📦 Starting upload to: {repo_id}\n")

# Files to upload
files_to_upload = [
    "best_model.pth",
    "gcn_model.pth",
    "transe_model.pth",
    "distmult_model.pth",
    "complex_model.pth",
    "graphsage_model.pth",
    "model_metadata.json",
    "best_model_info.json"
]

# Upload each file
for filename in files_to_upload:
    file_path = os.path.join(models_dir, filename)
    
    if os.path.exists(file_path):
        try:
            print(f"📤 Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"  ✓ {filename} uploaded successfully")
        except Exception as e:
            print(f"  ✗ Error uploading {filename}: {e}")
    else:
        print(f"  ⚠ {filename} not found at {file_path}")

print("\n✅ Upload complete!")
print(f"🔗 View your models at: https://huggingface.co/{repo_id}")
```

---

## Step 6: Create README.md for Model Card

Create a `README.md` file in your models directory:

```markdown
---
library_name: pytorch
tags:
- graph-neural-network
- knowledge-graph
- agricultural-ai
- crop-recommendation
- gcn
- graph-embeddings
license: mit
datasets:
- ugandan-agricultural-data
---

# Agricultural AI Graph Embedding Models

Graph neural network models trained on Ugandan agricultural knowledge graph for crop recommendation.

## Model Overview

This repository contains multiple graph embedding models trained on an agricultural knowledge graph with 175,318 triples representing crop-soil-climate relationships.

## Models Included

### Best Model: GCN (Graph Convolutional Network)
- **File**: `best_model.pth`
- **Accuracy**: 87.神经28%
- **F1-Score**: 85.71%
- **ROC-AUC**: 96.90%
- **Embedding Dimension**: 100
- **Entities**: 2,513
- **Relations**: 15

### Individual Models
1. **GCN Model** (`gcn_model.pth`) - Best performing
2. **TransE Model** (`transe_model.pth`) - Translation-based
3. **DistMult Model** (`distmult_model.pth`) - Bilinear
4. **ComplEx Model** (`complex_model.pth`) - Complex embeddings
5. **GraphSAGE Model** (`graphsage_model.pth`) - Sampling-based

## Model Metadata

The `model_metadata.json` file contains:
- Entity to ID mappings (2,513 entities)
- Relation to ID mappings (15 relations)
- ID to entity mappings
- Model configuration parameters

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/agricultural-ai-graph-models",
    filename="best_model.pth"
)

# Download metadata
metadata_path = hf_hub_download(
    repo_id="YOUR_USERNAME/agricultural-ai-graph-models",
    filename="model_metadata.json"
)

# Load model
model = YourGCNModel(...)  # Initialize with architecture
model.load_state_dict(torch.load(model_path))
```

## Training Data

- **Knowledge Graph**: 175,318 triples
- **Dataset**: Ugandan agricultural data
- **Literature**: 52 research papers
- **Crops**: 8 major crops (maize, rice, beans, cassava, sweet potato, banana, coffee, cotton)

## Application

Used in production for agricultural crop recommendations based on:
- Soil properties (pH, organic matter, nutrients)
- Climate conditions (temperature, rainfall)
- Knowledge graph embeddings

## Citation

If you use these models, please cite:

```bibtex
@misc{agricultural-ai-graph-models,
  title={Agricultural AI Graph Embedding Models for Crop Recommendation},
  author={Your Name},
  year={2025},
  publisher={Hugging Face}
}
```

## License

MIT License
```

Save this as `deployment/models/README.md`

---

## Step 7: Upload README and Complete Documentation

Add README to upload script:

```python
# Add to files_to_upload list:
files_to_upload.append("README.md")
```

Or upload separately:

```python
api.upload_file(
    path_or_fileobj="deployment/models/README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model"
)
```

---

## Step 8: Run the Upload Script

```bash
cd deployment
python upload_to_huggingface.py
```

**Note**: Large files (>5GB) may take time. The script will show progress.

---

## Step 9: Verify Upload

1. Visit your repository: `https://huggingface.co/YOUR_USERNAME/agricultural-ai-graph-models`
2. Check that all files are present
3. Verify file sizes match originals

---

## Alternative: Using Git LFS (For Large Files)

If files are very large (>100MB), use Git LFS:

```bash
# Install Git LFS
git lfs install

# Clone your Hugging Face repo
git clone https://huggingface.co/YOUR_USERNAME/agricultural-ai-graph-models

cd agricultural-ai-graph-models

# Track .pth files with LFS
git lfs track "*.pth"

# Copy files
cp ../deployment/models/*.pth .
cp ../deployment/models/*.json .
cp ../deployment/models/README.md .

# Commit and push
git add .
git commit -m "Upload agricultural AI graph models"
git push
```

---

## Step 10: Download Models Later

### Using Python

```python
from huggingface_hub import hf_hub_download

# Download specific file
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/agricultural-ai-graph-models",
    filename="best_model.pth",
    cache_dir="./models"
)

# Download all files
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YOUR_USERNAME/agricultural-ai-graph-models",
    local_dir="./downloaded_models"
)
```

### Using Command Line

```bash
huggingface-cli download YOUR_USERNAME/agricultural-ai-graph-models best_model.pth
```

---

## Troubleshooting

### Issue: "Token not found"
**Solution**: Run `huggingface-cli login` and enter your token

### Issue: "Repository not found"
**Solution**: Make sure you created the repository first on huggingface.co

### Issue: "File too large"
**Solution**: Use Git LFS for files >100MB. Hugging Face supports up to 50GB with LFS.

### Issue: Upload timeout
**Solution**: Large files may take time. Use `resume_download` in API or upload via Git LFS.

---

## File Size Estimation

Before uploading, check file sizes:

```bash
cd deployment/models
du -h *.pth *.json
```

Typical sizes:
- Model files (.pth): 5-50 MB each
- Metadata (.json): Can be large if entity mappings are extensive
- Total: Likely 100-500 MB

---

## Best Practices

1. **Version Tagging**: Tag releases on Hugging Face for version control
2. **Documentation**: Always include comprehensive README
3. **Privacy**: Set repository to private if models contain sensitive data
4. **Testing**: Test downloading and loading models after upload
5. **Metadata**: Include model performance metrics in README

---

## Quick Reference Commands

```bash
# Login
huggingface-cli login

# Create repository
huggingface-cli repo create agricultural-ai-graph-models --type model

# Download file
huggingface-cli download USERNAME/agricultural-ai-graph-models best_model.pth

# List files in repo
huggingface-cli repo info USERNAME/agricultural-ai-graph-models
```

---

## Next Steps After Upload

1. Share your model repository link
2. Create example notebooks demonstrating usage
3. Add inference code examples
4. Document the API integration
5. Add evaluation metrics visualization

---

## Support

- Hugging Face Docs: https://huggingface.co/docs/hub/uploading
- Hugging Face Discord: https://huggingface.co/join/discord

---

**Ready to completed? Run the upload script and share your models with the community!**

