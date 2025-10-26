"""
Phase 7, Cell 1: LLM Fine-tuning Setup and Data Preparation
This cell sets up the environment for fine-tuning a pretrained language model on agricultural data
"""

# Install additional packages for LLM fine-tuning
!pip install transformers[torch] datasets accelerate evaluate rouge-score
!pip install wandb  # Optional: for experiment tracking
!pip install bitsandbytes  # For memory optimization

# Import libraries
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset as HFDataset
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up paths (following your existing structure)
PROJECT_ROOT = '/content/drive/MyDrive/Final'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'trained_models')
LLM_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'llm_models')

# Create LLM models directory
os.makedirs(LLM_DIR, exist_ok=True)

print("✅ LLM Fine-tuning environment setup complete!")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"LLM models directory: {LLM_DIR}")

# Check available data
print("\n📊 Available Data Sources:")
data_files = [
    'unified_knowledge_graph.json',
    'dataset_triples.json', 
    'complete_literature_triples.json',
    'ugandan_data_cleaned.csv'
]

for file in data_files:
    file_path = os.path.join(DATA_DIR, file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"✅ {file}: {size:.1f} MB")
    else:
        print(f"❌ {file}: Not found")

print("\n🔧 PyTorch and Transformers versions:")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
