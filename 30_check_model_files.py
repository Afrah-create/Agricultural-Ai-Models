# Check Fine-tuned Model Files
# Run this first to see what files are available

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_model_files():
    """Check what files are available in the fine-tuned model directory"""
    
    print("🔍 Checking Fine-tuned Model Files")
    print("=" * 50)
    
    model_path = '/content/drive/MyDrive/Final/data/processed/llm_models/quick_fine_tuned_fast'
    
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return
    
    print(f"📁 Model directory: {model_path}")
    
    # List all files in the directory
    files = os.listdir(model_path)
    print(f"\n📄 Files in model directory:")
    for file in files:
        file_path = os.path.join(model_path, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  📁 {file} (directory)")
    
    # Check for required files
    required_files = ['config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']
    optional_files = ['pytorch_model.bin', 'model.safetensors', 'training_args.bin']
    
    print(f"\n🔍 Checking required files:")
    for file in required_files:
        if file in files:
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
    
    print(f"\n🔍 Checking optional files:")
    for file in optional_files:
        if file in files:
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️ {file} (not found)")
    
    # Try to load the model
    print(f"\n🔄 Attempting to load model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("✅ Tokenizer loaded successfully!")
        
        # Check tokenizer info
        print(f"📊 Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"📊 Tokenizer model max length: {tokenizer.model_max_length}")
        
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        print("✅ Model loaded successfully!")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        # Test a simple generation
        print(f"\n🧪 Testing model generation...")
        test_input = "Maize grows best in"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=30,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test prompt: {test_input}")
        print(f"Model response: {response}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 This might be due to missing model weights or configuration issues")

if __name__ == "__main__":
    check_model_files()
