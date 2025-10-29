#!/usr/bin/env python3
"""
Upload Agricultural AI Models to Hugging Face Hub

Usage:
    export HF_TOKEN="your_huggingface_token"
    python upload_to_huggingface.py

Or:
    python upload_to_huggingface.py --token your_token

Requirements:
    pip install huggingface_hub
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("❌ Error: huggingface_hub not installed")
    print("   Please install it: pip install huggingface_hub")
    sys.exit(1)

# Configuration
GRAPH_MODELS_REPO_ID = os.getenv("HF_GRAPH_REPO", "Awongo/soil-crop-recommendation-model")
FINETUNED_REPO_ID = os.getenv("HF_FINETUNED_REPO", "Awongo/agricultural-llm-finetuned")
GRAPH_MODELS_DIR = "deployment/models"
FINETUNED_MODEL_DIR = "deployment/quick_fine_tuned_fast"

def get_token(args=None):
    """Get Hugging Face token from environment or argument"""
    token = None
    if args and args.token:
        token = args.token
    else:
        token = os.getenv("HF_TOKEN")
    
    if not token:
        print("❌ Error: Hugging Face token not found")
        print("\n   Please provide token using one of these methods:")
        print("   1. Environment variable: export HF_TOKEN='your_token'")
        print("   2. Command line: python upload_to_huggingface.py --token your_token")
        print("\n   Get your token from: https://huggingface.co/settings/tokens")
        return None
    return token

def create_graph_models_readme(models_dir):
    """Create README.md for graph models if it doesn't exist"""
    readme_path = Path(models_dir) / "README.md"
    
    if readme_path.exists():
        print(f"   ✓ README.md already exists")
        return
    
    print(f"   Creating README.md...")
    
    readme_content = """---
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
- **Accuracy**: 87.28%
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
    repo_id="Awongo/soil-crop-recommendation-model",
    filename="best_model.pth"
)

# Download metadata
metadata_path = hf_hub_download(
    repo_id="Awongo/soil-crop-recommendation-model",
    filename="model_metadata.json"
)

# Load model (pseudo-code - adjust to your model architecture)
# model = GCNModel(num_entities=2513, num_relations=15, embedding_dim=100)
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
# model.eval()
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

```bibtex
@misc{agricultural-ai-graph-models,
  title={Agricultural AI Graph Embedding Models for Crop Recommendation},
  year={2025},
  publisher={Hugging Face}
}
```
"""
    
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_content)
    print(f"   ✓ README.md created at {readme_path}")

def create_finetuned_readme(finetuned_dir):
    """Create README.md for fine-tuned model if it doesn't exist"""
    readme_path = Path(finetuned_dir) / "README.md"
    
    if readme_path.exists():
        print(f"   ✓ README.md already exists")
        return
    
    print(f"   Creating README.md...")
    
    readme_content = """---
library_name: transformers
tags:
- agricultural-ai
- llm
- fine-tuned
- crop-recommendation
- dialogpt
- text-generation
license: mit
datasets:
- ugandan-agricultural-data
- agricultural-literature
base_model: microsoft/DialoGPT-small
---

# Agricultural AI Fine-Tuned LLM

Fine-tuned DialoGPT-small model for agricultural crop recommendation and expert analysis generation.

## Model Overview

This model is a fine-tuned version of Microsoft's DialoGPT-small, trained specifically on agricultural domain knowledge for generating crop recommendations and expert analysis.

## Training Details

- **Base Model**: microsoft/DialoGPT-small
- **Training Data**: Agricultural knowledge graph triples, literature reviews, Ugandan agricultural dataset
- **Training Approach**: Domain-specific fine-tuning
- **Purpose**: Generate contextual agricultural recommendations and expert analysis

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Awongo/agricultural-llm-finetuned")
model = AutoModelForCausalLM.from_pretrained("Awongo/agricultural-llm-finetuned")

# Generate recommendation
prompt = "Given soil pH 6.2, organic matter 3%, and temperature 24°C, recommend suitable crops:"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Integration

This model is used in the Agricultural AI system alongside:
- Graph Convolutional Networks (GCN) for entity embeddings
- Constraint-based reasoning engine
- Retrieval-Augmented Generation (RAG) pipeline

## Citation

```bibtex
@misc{agricultural-llm-finetuned,
  title={Agricultural AI Fine-Tuned Language Model for Crop Recommendation},
  year={2025},
  publisher={Hugging Face}
}
```
"""
    
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_content)
    print(f"   ✓ README.md created at {readme_path}")

def upload_folder(api, folder_path, repo_id, repo_type="model", ignore_patterns=None):
    """Upload a folder to Hugging Face Hub"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return False
    
    if not folder_path.is_dir():
        print(f"❌ Path is not a directory: {folder_path}")
        return False
    
    print(f"📤 Uploading folder: {folder_path}")
    print(f"   To repository: {repo_id}\n")
    
    try:
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type=repo_type,
            ignore_patterns=ignore_patterns
        )
        print(f"✅ Successfully uploaded to: {repo_id}")
        return True
    except HfHubHTTPError as e:
        error_message = str(e)
        if "404" in error_message or "not found" in error_message.lower():
            print(f"❌ Repository not found: {repo_id}")
            print(f"\n   Please create the repository first:")
            print(f"   1. Visit: https://huggingface.co/new")
            print(f"   2. Repository name: {repo_id.split('/')[-1]}")
            print(f"   3. Repository type: Model")
            print(f"   4. Or run: huggingface-cli repo create {repo_id.split('/')[-1]} --type model")
        elif "403" in error_message or "Forbidden" in error_message or "permissions" in error_message.lower():
            print(f"❌ Permission denied: {repo_id}")
            print(f"\n   Possible issues:")
            print(f"   1. Token doesn't have write permissions")
            print(f"   2. Repository doesn't exist - create it first at https://huggingface.co/new")
            print(f"   3. Token is invalid - check at https://huggingface.co/settings/tokens")
            print(f"\n   Error details: {error_message}")
        else:
            print(f"❌ Error uploading: {e}")
        return False
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        return False

def upload_graph_models(api, token):
    """Upload graph embedding models"""
    print("="*70)
    print("📦 Uploading Graph Embedding Models")
    print("="*70)
    
    models_path = Path(GRAPH_MODELS_DIR)
    if not models_path.exists():
        print(f"❌ Models directory not found: {GRAPH_MODELS_DIR}")
        print(f"   Current directory: {os.getcwd()}")
        return False
    
    # Create README if needed
    create_graph_models_readme(GRAPH_MODELS_DIR)
    
    # Ignore patterns for unnecessary files
    ignore_patterns = ["__pycache__", "*.pyc", ".git"]
    
    success = upload_folder(
        api=api,
        folder_path=models_path,
        repo_id=GRAPH_MODELS_REPO_ID,
        repo_type="model",
        ignore_patterns=ignore_patterns
    )
    
    if success:
        print(f"\n🔗 View graph models at:")
        print(f"   https://huggingface.co/{GRAPH_MODELS_REPO_ID}\n")
    
    return success

def upload_finetuned_model(api, token):
    """Upload fine-tuned LLM model"""
    print("="*70)
    print("📦 Uploading Fine-Tuned LLM Model")
    print("="*70)
    
    finetuned_path = Path(FINETUNED_MODEL_DIR)
    if not finetuned_path.exists():
        print(f"❌ Fine-tuned model directory not found: {FINETUNED_MODEL_DIR}")
        print(f"   Current directory: {os.getcwd()}")
        return False
    
    # Create README if needed
    create_finetuned_readme(FINETUNED_MODEL_DIR)
    
    # Upload only essential files from root directory (exclude checkpoints and logs)
    # This avoids UTF-8 encoding errors with some binary files
    essential_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "README.md"
    ]
    
    print("📤 Uploading essential model files...\n")
    
    uploaded_count = 0
    for filename in essential_files:
        file_path = finetuned_path / filename
        if file_path.exists():
            try:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  Uploading {filename} ({file_size:.2f} MB)...", end=" ", flush=True)
                
                # Read file in binary mode and upload
                with open(file_path, 'rb') as f:
                    api.upload_file(
                        path_or_fileobj=f,
                        path_in_repo=filename,
                        repo_id=FINETUNED_REPO_ID,
                        repo_type="model"
                    )
                
                print("✓")
                uploaded_count += 1
            except Exception as e:
                print(f"✗ Error: {e}")
        else:
            print(f"  ⚠ {filename} not found, skipping...")
    
    if uploaded_count > 0:
        print(f"\n✅ Successfully uploaded {uploaded_count} files")
        print(f"\n🔗 View fine-tuned model at:")
        print(f"   https://huggingface.co/{FINETUNED_REPO_ID}\n")
        return True
    else:
        print("\n❌ No files were uploaded")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    parser.add_argument("--graph-repo", type=str, help="Graph models repository ID")
    parser.add_argument("--finetuned-repo", type=str, help="Fine-tuned model repository ID")
    parser.add_argument("--graph-only", action="store_true", help="Upload only graph models")
    parser.add_argument("--finetuned-only", action="store_true", help="Upload only fine-tuned model")
    
    args = parser.parse_args()
    
    # Override repo IDs if provided via arguments Horror
    global GRAPH_MODELS_REPO_ID, FINETUNED_REPO_ID
    if args.graph_repo:
        GRAPH_MODELS_REPO_ID = args.graph_repo
    if args.finetuned_repo:
        FINETUNED_REPO_ID = args.finetuned_repo
    
    print("="*70)
    print("🚀 Agricultural AI Models Upload to Hugging Face")
    print("="*70)
    print(f"\nGraph Models Repository: {GRAPH_MODELS_REPO_ID}")
    print(f"Fine-Tuned Model Repository: {FINETUNED_REPO_ID}\n")
    
    # Get token
    token = get_token(args)
    if not token:
        sys.exit(1)
    
    # Initialize API with token
    api = HfApi(token=token)
    
    # Verify token works
    try:
        user_info = api.whoami()
        print(f"✓ Authenticated as: {user_info.get('name', 'Unknown')}\n")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Please check your token is valid")
        sys.exit(1)
    
    # Upload models
    results = {}
    
    if not args.finetuned_only:
        results['graph'] = upload_graph_models(api, token)
    
    if not args.graph_only:
        results['finetuned'] = upload_finetuned_model(api, token)
    
    # Summary
    print("="*70)
    print("📊 Upload Summary")
    print("="*70)
    
    if results.get('graph'):
        print(f"✅ Graph models uploaded: {GRAPH_MODELS_REPO_ID}")
    else:
        print(f"❌ Graph models upload failed")
    
    if results.get('finetuned'):
        print(f"✅ Fine-tuned model uploaded: {FINETUNED_REPO_ID}")
    else:
        print(f"❌ Fine-tuned model upload failed")
    
    print("="*70)
    
    if all(results.values()):
        print("✅ All uploads completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some uploads failed. Please check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
