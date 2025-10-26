# Test Fine-tuned Model Integration
# Simple test to verify the model works

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_finetuned_model():
    """Test the fine-tuned model integration"""
    
    print("🧪 Testing Fine-tuned Model Integration")
    print("=" * 50)
    
    # Check if model files exist
    model_path = os.path.join('..', 'quick_fine_tuned_fast')
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    
    print(f"✅ Model path found: {model_path}")
    
    # List model files
    files = os.listdir(model_path)
    print(f"📄 Model files: {len(files)} files")
    
    # Check for required files
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    missing_files = [f for f in required_files if f not in files]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    
    try:
        # Load model
        print("🔄 Loading fine-tuned model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        # Test generation
        print("\n🧪 Testing model generation...")
        test_prompts = [
            "Maize grows best in",
            "Rice requires",
            "Soil pH should be"
        ]
        
        for prompt in test_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                print(f"Prompt: {prompt}")
                print(f"Response: {response}")
                print()
                
            except Exception as e:
                print(f"Error with prompt '{prompt}': {e}")
        
        print("✅ Fine-tuned model integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    success = test_finetuned_model()
    if success:
        print("\n🎉 Integration successful! Your fine-tuned model is ready to use.")
    else:
        print("\n❌ Integration failed. Please check the model files.")
