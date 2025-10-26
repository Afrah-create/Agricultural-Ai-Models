# Quick LLM Fine-tuning with Optimized Settings
# This script reduces training time from 82 hours to ~20-30 hours

import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import json

# Optimized model selection
MODEL_CONFIGS = {
    'fast': {
        'model_name': 'distilbert/distilgpt2',
        'description': 'Distilled GPT-2 (Fastest)',
        'memory_requirement': 'Low (~500MB)',
        'estimated_time': '15-20 hours'
    },
    'balanced': {
        'model_name': 'gpt2',
        'description': 'Standard GPT-2 (Balanced)',
        'memory_requirement': 'Medium (~1.5GB)',
        'estimated_time': '25-35 hours'
    },
    'quality': {
        'model_name': 'microsoft/DialoGPT-small',
        'description': 'DialoGPT (Current)',
        'memory_requirement': 'Medium (~1.2GB)',
        'estimated_time': '40-50 hours'
    }
}

def optimize_dataset(text_samples, max_samples=5000):
    """Optimize dataset for faster training while maintaining quality"""
    print(f"📊 Original dataset: {len(text_samples)} samples")
    
    # Prioritize by sample type and quality
    priority_samples = []
    
    # Count samples by type
    type_counts = {}
    for sample in text_samples:
        sample_type = sample.get('type', 'unknown')
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
    
    print(f"📈 Sample distribution:")
    for sample_type, count in type_counts.items():
        print(f"  - {sample_type}: {count}")
    
    # 1. Recommendations (highest priority) - take up to 40%
    recommendations = [s for s in text_samples if s.get('type') == 'recommendation']
    priority_samples.extend(recommendations[:int(max_samples * 0.4)])
    
    # 2. Q&A pairs (high value) - take up to 30%
    qa_pairs = [s for s in text_samples if s.get('type') == 'qa_pair']
    priority_samples.extend(qa_pairs[:int(max_samples * 0.3)])
    
    # 3. Research findings (good value) - take up to 20%
    research = [s for s in text_samples if s.get('type') == 'research_finding']
    priority_samples.extend(research[:int(max_samples * 0.2)])
    
    # 4. Other types - take remaining 10%
    other_samples = [s for s in text_samples if s.get('type') not in ['recommendation', 'qa_pair', 'research_finding']]
    priority_samples.extend(other_samples[:int(max_samples * 0.1)])
    
    # If we still don't have enough, fill with random samples
    if len(priority_samples) < max_samples:
        remaining_needed = max_samples - len(priority_samples)
        remaining_samples = [s for s in text_samples if s not in priority_samples]
        priority_samples.extend(remaining_samples[:remaining_needed])
    
    print(f"✅ Optimized dataset: {len(priority_samples)} samples")
    return priority_samples

def setup_fast_training(model_choice='fast'):
    """Setup optimized training configuration"""
    
    model_config = MODEL_CONFIGS[model_choice]
    print(f"🚀 Selected model: {model_config['model_name']}")
    print(f"📝 Description: {model_config['description']}")
    print(f"💾 Memory: {model_config['memory_requirement']}")
    print(f"⏱️ Estimated time: {model_config['estimated_time']}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    model = AutoModelForCausalLM.from_pretrained(model_config['model_name'])
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, model_config

def create_optimized_training_args(output_dir, model_config):
    """Create optimized training arguments for faster training"""
    
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Reduced training time
        num_train_epochs=1,  # Single epoch for speed
        per_device_train_batch_size=4,  # Larger batch size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # Reduced accumulation
        
        # Optimized learning rate
        learning_rate=2e-4,  # Higher for faster convergence
        warmup_steps=50,  # Reduced warmup
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=500,  # Less frequent evaluation
        save_steps=1000,  # Less frequent saving
        save_total_limit=2,  # Keep only 2 checkpoints
        
        # Performance optimizations
        fp16=torch.cuda.is_available(),  # Mixed precision
        dataloader_num_workers=2,  # Parallel data loading
        dataloader_pin_memory=True,
        
        # Logging
        logging_steps=100,
        logging_dir=os.path.join(output_dir, 'logs'),
        
        # Early stopping
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Disable unnecessary features
        prediction_loss_only=True,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb for speed
    )

def quick_fine_tune():
    """Execute quick fine-tuning with optimized settings"""
    
    print("🚀 Starting Quick LLM Fine-tuning...")
    print("=" * 50)
    
    # Setup paths - Updated to match your actual Google Drive structure
    LLM_DIR = '/content/drive/MyDrive/Final/data/processed/llm_models'
    DATA_DIR = '/content/drive/MyDrive/Final/data/processed'
    
    # Choose model (change this to 'fast', 'balanced', or 'quality')
    model_choice = 'fast'  # Change this for different speed/quality trade-offs
    
    # Check for dataset file in multiple possible locations
    possible_dataset_paths = [
        os.path.join(DATA_DIR, 'agricultural_text_dataset.json'),
        os.path.join(DATA_DIR, 'text_samples.json'),
        os.path.join('/content/drive/MyDrive/Final/data/processed', 'agricultural_text_dataset.json'),
        os.path.join('/content/drive/MyDrive/Final/data/processed', 'text_samples.json'),
        os.path.join('/content/drive/MyDrive/Final', 'agricultural_text_dataset.json'),
        os.path.join('/content/drive/MyDrive/Final', 'text_samples.json')
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✅ Found dataset at: {path}")
            break
    
    if not dataset_path:
        print("❌ Dataset file not found in any expected location!")
        print("Expected locations:")
        for path in possible_dataset_paths:
            print(f"  - {path}")
        print("\nPlease check if the dataset was created successfully in the previous steps.")
        return
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text_samples = json.load(f)
        
        # Optimize dataset
        optimized_samples = optimize_dataset(text_samples, max_samples=5000)
        
        # Setup model
        model, tokenizer, model_config = setup_fast_training(model_choice)
        
        # Tokenize dataset
        print("🔄 Tokenizing dataset...")
        tokenized_samples = []
        for sample in optimized_samples:
            text = sample.get('text', '')
            if text:
                tokens = tokenizer(text, truncation=True, padding='max_length', max_length=256)
                tokenized_samples.append(tokens)
        
        # Split dataset
        train_size = int(0.8 * len(tokenized_samples))
        train_data = tokenized_samples[:train_size]
        val_data = tokenized_samples[train_size:]
        
        # Create HuggingFace datasets
        train_dataset = HFDataset.from_list(train_data)
        val_dataset = HFDataset.from_list(val_data)
        
        # Setup training
        output_dir = os.path.join(LLM_DIR, f'quick_fine_tuned_{model_choice}')
        training_args = create_optimized_training_args(output_dir, model_config)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        print(f"\n⚙️ Training Configuration:")
        print(f"Model: {model_config['model_name']}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Estimated time: {model_config['estimated_time']}")
        
        # Start training
        print(f"\n🎯 Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ Quick fine-tuning complete!")
        print(f"Model saved to: {output_dir}")
        
    else:
        print("❌ Dataset file not found!")

if __name__ == "__main__":
    quick_fine_tune()
