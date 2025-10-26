# Improved Quick LLM Fine-tuning Script
# This version uses more samples for meaningful training

import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import json

def create_balanced_dataset(text_samples, target_samples=3000):
    """Create a balanced dataset with meaningful sample distribution"""
    print(f"📊 Original dataset: {len(text_samples)} samples")
    
    # Count samples by type
    type_counts = {}
    for sample in text_samples:
        sample_type = sample.get('type', 'unknown')
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
    
    print(f"📈 Sample distribution:")
    for sample_type, count in type_counts.items():
        print(f"  - {sample_type}: {count}")
    
    # Create balanced dataset
    balanced_samples = []
    
    # Calculate how many samples to take from each type
    total_available = sum(type_counts.values())
    if total_available == 0:
        print("❌ No samples found!")
        return []
    
    # Take samples proportionally but cap at reasonable amounts
    for sample_type, count in type_counts.items():
        if count > 0:
            # Take up to 30% of target samples from each type, but at least 50 samples
            max_from_type = max(50, int(target_samples * 0.3))
            samples_to_take = min(count, max_from_type)
            
            type_samples = [s for s in text_samples if s.get('type') == sample_type]
            balanced_samples.extend(type_samples[:samples_to_take])
            print(f"  ✅ Added {samples_to_take} {sample_type} samples")
    
    # If we still need more samples, fill with random ones
    if len(balanced_samples) < target_samples:
        remaining_needed = target_samples - len(balanced_samples)
        remaining_samples = [s for s in text_samples if s not in balanced_samples]
        balanced_samples.extend(remaining_samples[:remaining_needed])
        print(f"  ✅ Added {min(remaining_needed, len(remaining_samples))} additional samples")
    
    print(f"✅ Balanced dataset: {len(balanced_samples)} samples")
    return balanced_samples

def run_improved_training():
    """Run improved quick training with better dataset balance"""
    
    print("🚀 Starting Improved Quick LLM Fine-tuning...")
    print("=" * 60)
    
    # Setup paths
    LLM_DIR = '/content/drive/MyDrive/Final/data/processed/llm_models'
    DATA_DIR = '/content/drive/MyDrive/Final/data/processed'
    
    # Check for dataset file
    dataset_path = os.path.join(DATA_DIR, 'agricultural_text_dataset.json')
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset file not found!")
        return
    
    print(f"✅ Found dataset at: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text_samples = json.load(f)
    
    # Create balanced dataset
    balanced_samples = create_balanced_dataset(text_samples, target_samples=3000)
    
    if len(balanced_samples) < 100:
        print("❌ Not enough samples for meaningful training!")
        return
    
    # Load model and tokenizer
    print("\n🔄 Loading DistilGPT-2 model...")
    model_name = 'distilbert/distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded: {model_name}")
    print(f"📊 Model parameters: {model.num_parameters():,}")
    
    # Tokenize dataset
    print("\n🔄 Tokenizing dataset...")
    tokenized_samples = []
    
    for i, sample in enumerate(balanced_samples):
        if i % 500 == 0:
            print(f"  Processing sample {i}/{len(balanced_samples)}")
        
        text = sample.get('text', '')
        if text and len(text.strip()) > 10:  # Only use meaningful text
            tokens = tokenizer(
                text, 
                truncation=True, 
                padding='max_length', 
                max_length=256
            )
            tokenized_samples.append(tokens)
    
    print(f"✅ Tokenized {len(tokenized_samples)} samples")
    
    # Split dataset
    train_size = int(0.8 * len(tokenized_samples))
    train_data = tokenized_samples[:train_size]
    val_data = tokenized_samples[train_size:]
    
    # Create HuggingFace datasets
    train_dataset = HFDataset.from_list(train_data)
    val_dataset = HFDataset.from_list(val_data)
    
    # Training configuration
    output_dir = os.path.join(LLM_DIR, 'improved_quick_model')
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training settings
        num_train_epochs=2,  # 2 epochs for better learning
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        
        # Learning rate
        learning_rate=1e-4,
        warmup_steps=100,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=3,
        
        # Performance optimizations
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        
        # Logging
        logging_steps=50,
        logging_dir=os.path.join(output_dir, 'logs'),
        
        # Early stopping
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Other settings
        prediction_loss_only=True,
        remove_unused_columns=False,
        report_to=None,
    )
    
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
    print(f"Model: {model_name}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Estimated time: 2-4 hours")
    
    # Start training
    print(f"\n🎯 Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Improved training complete!")
    print(f"Model saved to: {output_dir}")
    
    # Test the model
    print(f"\n🧪 Testing the model...")
    test_prompts = [
        "Maize grows best in",
        "Rice requires",
        "Beans are good for",
        "Soil pH should be"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")
        print()

if __name__ == "__main__":
    run_improved_training()
