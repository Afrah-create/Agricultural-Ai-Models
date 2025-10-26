"""
Phase 7, Cell 4: LLM Fine-tuning Execution
This cell executes the fine-tuning process and saves the trained model
"""

import torch
from transformers import Trainer
import os
import json
from datetime import datetime
import time

print("🚀 Starting LLM Fine-tuning Process...")

# Check if we have everything ready
if train_hf_dataset is None or val_hf_dataset is None:
    print("❌ Datasets not ready. Please run previous cells first.")
else:
    print("✅ All components ready for fine-tuning!")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=val_hf_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(f"\n🎯 Starting training with {len(train_hf_dataset):,} training samples...")
    print(f"Training will take approximately {len(train_hf_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs // 100} minutes")
    
    # Start training
    start_time = time.time()
    
    try:
        # Train the model
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")
        
        # Save the fine-tuned model
        print("\n💾 Saving fine-tuned model...")
        
        # Save model and tokenizer
        model_save_path = os.path.join(LLM_DIR, 'agricultural_llm')
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        print(f"✅ Model saved to: {model_save_path}")
        
        # Save training metadata
        training_metadata = {
            'model_name': model_config['model_name'],
            'training_samples': len(train_hf_dataset),
            'validation_samples': len(val_hf_dataset),
            'training_time_minutes': training_time / 60,
            'training_args': {
                'num_train_epochs': training_args.num_train_epochs,
                'per_device_train_batch_size': training_args.per_device_train_batch_size,
                'learning_rate': training_args.learning_rate,
                'fp16': training_args.fp16
            },
            'timestamp': datetime.now().isoformat(),
            'model_path': model_save_path
        }
        
        metadata_path = os.path.join(LLM_DIR, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        print(f"✅ Training metadata saved to: {metadata_path}")
        
        # Test the fine-tuned model
        print("\n🧪 Testing fine-tuned model...")
        
        # Load the fine-tuned model for testing
        test_model = AutoModelForCausalLM.from_pretrained(model_save_path)
        test_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        
        # Test prompts
        test_prompts = [
            "What crops are suitable for loamy soil?",
            "How much rainfall does maize need?",
            "What is the optimal pH for rice cultivation?",
            "Agricultural Recommendation:",
            "Based on soil conditions, I recommend"
        ]
        
        print("\n📝 Model Test Results:")
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt}")
            
            # Generate response
            inputs = test_tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = test_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=test_tokenizer.eos_token_id
                )
            
            response = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
        
        print(f"\n🎉 Fine-tuning process completed successfully!")
        print(f"Model saved to: {model_save_path}")
        print(f"Model size: {os.path.getsize(os.path.join(model_save_path, 'pytorch_model.bin')) / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This might be due to memory constraints or other issues.")
        print("Try reducing batch size or using a smaller model.")

# Alternative: Quick test without full training
print("\n🔬 Quick Model Test (without fine-tuning)...")

# Test the base model
test_prompt = "What crops are suitable for loamy soil?"
inputs = tokenizer.encode(test_prompt, return_tensors='pt')

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

base_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Base model response: {base_response}")

print("\n📊 Training Summary:")
print(f"Model: {model_config['model_name']}")
print(f"Training samples: {len(train_hf_dataset) if train_hf_dataset else 0}")
print(f"Model parameters: {model.num_parameters():,}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Training completed: {'Yes' if 'trainer' in locals() else 'No'}")

# Save model info for deployment
model_info = {
    'model_name': model_config['model_name'],
    'model_path': os.path.join(LLM_DIR, 'agricultural_llm'),
    'tokenizer_path': os.path.join(LLM_DIR, 'agricultural_llm'),
    'training_samples': len(train_hf_dataset) if train_hf_dataset else 0,
    'model_parameters': model.num_parameters(),
    'fine_tuned': 'trainer' in locals(),
    'timestamp': datetime.now().isoformat()
}

info_path = os.path.join(LLM_DIR, 'model_info.json')
with open(info_path, 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n💾 Model info saved to: {info_path}")
print("✅ LLM fine-tuning process complete!")
