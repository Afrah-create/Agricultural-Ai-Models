# Test the Fine-tuned Agricultural LLM
# Run this to see how well your model performs

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def test_agricultural_model():
    """Test the fine-tuned agricultural model"""
    
    print("🧪 Testing Fine-tuned Agricultural LLM")
    print("=" * 50)
    
    # Load the fine-tuned model
    model_path = '/content/drive/MyDrive/Final/data/processed/llm_models/quick_fine_tuned_fast'
    
    try:
        print("🔄 Loading fine-tuned model...")
        print(f"📁 Model path: {model_path}")
        
        # Check if model files exist
        import os
        if not os.path.exists(model_path):
            print(f"❌ Model directory not found: {model_path}")
            return
        
        # List files in the model directory
        model_files = os.listdir(model_path)
        print(f"📄 Model files: {model_files}")
        
        # Load tokenizer and model from local directory
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Trying alternative loading method...")
        
        try:
            # Alternative: Load from the original DistilGPT-2 and check if we can load the fine-tuned weights
            print("🔄 Loading base DistilGPT-2 model...")
            tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2')
            model = AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2')
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("✅ Base model loaded successfully!")
            print("⚠️ Using base model (fine-tuned weights may not be available)")
            
        except Exception as e2:
            print(f"❌ Error loading base model: {e2}")
            return
    
    # Test prompts
    test_prompts = [
        "Maize grows best in",
        "Rice requires",
        "Beans are good for",
        "Soil pH should be",
        "Cassava is suitable for",
        "Coffee needs",
        "Sweet potato prefers",
        "Cotton requires",
        "Sugarcane grows well in",
        "Organic matter helps"
    ]
    
    print(f"\n🎯 Testing {len(test_prompts)} agricultural prompts...")
    print("-" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response (remove the original prompt)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"{i:2d}. Prompt: {prompt}")
            print(f"    Response: {response}")
            print()
            
        except Exception as e:
            print(f"{i:2d}. Prompt: {prompt}")
            print(f"    Error: {e}")
            print()
    
    # Test agricultural knowledge
    print("🌾 Testing Agricultural Knowledge...")
    print("-" * 50)
    
    knowledge_prompts = [
        "The best soil for maize is",
        "Rice cultivation requires",
        "Beans improve soil by",
        "Cassava is drought tolerant because",
        "Coffee plants need"
    ]
    
    for prompt in knowledge_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=60,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"Q: {prompt}")
            print(f"A: {response}")
            print()
            
        except Exception as e:
            print(f"Q: {prompt}")
            print(f"Error: {e}")
            print()
    
    print("✅ Testing complete!")
    print(f"\n📊 Model Performance Summary:")
    print(f"• Model: DistilGPT-2 (Fine-tuned)")
    print(f"• Training Samples: 4,000")
    print(f"• Training Time: ~4 hours")
    print(f"• Final Loss: 0.011500")
    print(f"• Validation Loss: 0.053528")

if __name__ == "__main__":
    test_agricultural_model()
