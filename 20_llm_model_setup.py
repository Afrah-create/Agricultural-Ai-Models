"""
Phase 7, Cell 3: LLM Model Selection and Fine-tuning Configuration
This cell sets up the pretrained model and fine-tuning configuration
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset as HFDataset
import json
import os
from datetime import datetime

print("🤖 Setting up LLM Model and Fine-tuning Configuration...")

# Model selection - using smaller models suitable for fine-tuning
MODEL_OPTIONS = {
    'distilgpt2': {
        'model_name': 'distilbert/distilgpt2',
        'description': 'Small GPT-2 model, fast training',
        'memory_requirement': 'Low (~500MB)',
        'suitable_for': 'Quick experiments'
    },
    'gpt2': {
        'model_name': 'gpt2',
        'description': 'Standard GPT-2 model',
        'memory_requirement': 'Medium (~1.5GB)',
        'suitable_for': 'Balanced performance'
    },
    'microsoft_dialogpt': {
        'model_name': 'microsoft/DialoGPT-small',
        'description': 'Dialogue-optimized GPT model',
        'memory_requirement': 'Medium (~1.2GB)',
        'suitable_for': 'Conversational responses'
    }
}

# Select model (change this to try different models)
SELECTED_MODEL = 'microsoft_dialogpt'  # Good for conversational agricultural advice
model_config = MODEL_OPTIONS[SELECTED_MODEL]

print(f"Selected model: {model_config['model_name']}")
print(f"Description: {model_config['description']}")
print(f"Memory requirement: {model_config['memory_requirement']}")

# Load tokenizer and model
print(f"\n🔄 Loading {model_config['model_name']}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name'],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    print("✅ Model loaded successfully!")
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Falling back to DistilGPT-2...")
    
    # Fallback to smaller model
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2')
    model_config['model_name'] = 'distilbert/distilgpt2'

# Load the agricultural text dataset
print("\n📚 Loading agricultural text dataset...")
dataset_path = os.path.join(DATA_DIR, 'agricultural_text_dataset.json')

if os.path.exists(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text_samples = json.load(f)
    print(f"✅ Loaded {len(text_samples):,} text samples")
else:
    print("❌ Dataset not found. Please run the previous cell first.")
    text_samples = []

# Prepare dataset for training
def tokenize_texts(texts, tokenizer, max_length=512):
    """Tokenize texts and return HuggingFace dataset format"""
    tokenized_data = []
    
    for text in texts:
        # Tokenize text
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        tokenized_data.append({
            'input_ids': encoding['input_ids'].flatten().tolist(),
            'attention_mask': encoding['attention_mask'].flatten().tolist(),
            'labels': encoding['input_ids'].flatten().tolist()
        })
    
    return tokenized_data

# Create dataset
if text_samples:
    # Extract just the text for training
    texts = [sample['text'] for sample in text_samples]
    
    # Split into train/validation
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    print(f"Training samples: {len(train_texts):,}")
    print(f"Validation samples: {len(val_texts):,}")
    
    # Tokenize training and validation texts
    print("🔄 Tokenizing training texts...")
    train_tokenized = tokenize_texts(train_texts, tokenizer)
    
    print("🔄 Tokenizing validation texts...")
    val_tokenized = tokenize_texts(val_texts, tokenizer)
    
    # Create HuggingFace datasets
    train_hf_dataset = HFDataset.from_list(train_tokenized)
    val_hf_dataset = HFDataset.from_list(val_tokenized)
    
    print("✅ Datasets prepared for training!")
    
else:
    print("❌ No text samples available for training")
    train_hf_dataset = None
    val_hf_dataset = None

# Training configuration
training_args = TrainingArguments(
    output_dir=os.path.join(LLM_DIR, 'fine_tuned_model'),
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust based on your data size
    per_device_train_batch_size=2,  # Small batch size for memory efficiency
    per_device_eval_batch_size=2,
    warmup_steps=100,
    learning_rate=5e-5,  # Lower learning rate for fine-tuning
    logging_steps=50,
    eval_strategy="steps",  # Updated parameter name
    eval_steps=200,
    save_steps=500,
    save_total_limit=3,
    prediction_loss_only=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
    gradient_accumulation_steps=4,  # Accumulate gradients for effective larger batch size
    report_to=None,  # Disable wandb for now
)

print("\n⚙️ Training Configuration:")
print(f"Output directory: {training_args.output_dir}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Mixed precision: {training_args.fp16}")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked
)

print("\n✅ LLM fine-tuning setup complete!")
print(f"Model: {model_config['model_name']}")
print(f"Training samples: {len(train_hf_dataset) if train_hf_dataset else 0}")
print(f"Validation samples: {len(val_hf_dataset) if val_hf_dataset else 0}")
print(f"Ready for fine-tuning: {'Yes' if train_hf_dataset else 'No'}")
