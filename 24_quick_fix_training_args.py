"""
QUICK FIX CELL - Run this to resolve the TrainingArguments error
This cell fixes the deprecated parameter issue
"""

# Quick fix for TrainingArguments error
print("🔧 Applying quick fix for TrainingArguments...")

# Updated training configuration with correct parameter names
training_args = TrainingArguments(
    output_dir=os.path.join(LLM_DIR, 'fine_tuned_model'),
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust based on your data size
    per_device_train_batch_size=2,  # Small batch size for memory efficiency
    per_device_eval_batch_size=2,
    warmup_steps=100,
    learning_rate=5e-5,  # Lower learning rate for fine-tuning
    logging_steps=50,
    eval_strategy="steps",  # Updated parameter name (was evaluation_strategy)
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

print("✅ Training configuration fixed!")
print("\n⚙️ Training Configuration:")
print(f"Output directory: {training_args.output_dir}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Mixed precision: {training_args.fp16}")
print(f"Evaluation strategy: {training_args.eval_strategy}")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked
)

print("\n✅ All components ready for training!")
print(f"Model: {model_config['model_name']}")
print(f"Training samples: {len(train_hf_dataset) if 'train_hf_dataset' in locals() else 0}")
print(f"Validation samples: {len(val_hf_dataset) if 'val_hf_dataset' in locals() else 0}")
print("Ready for fine-tuning: Yes")

print("\n🎯 Next step: Run the training execution cell!")
