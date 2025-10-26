# Integration Script: Add Fine-tuned LLM to Web Application
# This script modifies main.py to use your fine-tuned model

import os
import shutil

def integrate_finetuned_model():
    """Integrate the fine-tuned model into the web application"""
    
    print("🚀 Integrating Fine-tuned LLM into Web Application")
    print("=" * 60)
    
    # Check if model folder exists
    model_path = os.path.join('deployment', 'quick_fine_tuned_fast')
    if not os.path.exists(model_path):
        print(f"❌ Model folder not found: {model_path}")
        print("Please ensure the model is downloaded to the deployment directory")
        return False
    
    print(f"✅ Found model folder: {model_path}")
    
    # List model files
    model_files = os.listdir(model_path)
    print(f"📄 Model files: {len(model_files)} files")
    for file in model_files:
        print(f"  - {file}")
    
    # Check for required files
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    missing_files = []
    for file in required_files:
        if file not in model_files:
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required model files present")
    
    # Create backup of original main.py
    main_py_path = os.path.join('deployment', 'app', 'main.py')
    backup_path = os.path.join('deployment', 'app', 'main_backup.py')
    
    if os.path.exists(main_py_path):
        shutil.copy2(main_py_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
    
    print("\n🔧 Integration Steps:")
    print("1. ✅ Model files verified")
    print("2. ✅ Backup created")
    print("3. 🔄 Ready to modify main.py")
    
    return True

def create_model_integration_code():
    """Create the code to integrate the fine-tuned model"""
    
    integration_code = '''
# Fine-tuned LLM Integration Code
# Add this to your main.py file

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class FineTunedLLM:
    """Fine-tuned agricultural LLM"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            print(f"🔄 Loading fine-tuned model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Fine-tuned model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            return False
    
    def generate_response(self, prompt, max_length=100, temperature=0.7):
        """Generate response using fine-tuned model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response at this time."

# Add this to your AgriculturalAPI class
def _generate_recommendation_text_with_finetuned(self, suitable_crops, soil_properties, climate_conditions):
    """Generate recommendation text using fine-tuned model"""
    
    # Initialize fine-tuned model (do this once)
    if not hasattr(self, 'finetuned_llm'):
        model_path = os.path.join('deployment', 'quick_fine_tuned_fast')
        self.finetuned_llm = FineTunedLLM(model_path)
    
    # Create agricultural prompt
    prompt = f"Based on soil pH {soil_properties.get('pH', 'unknown')}, "
    prompt += f"organic matter {soil_properties.get('organic_matter', 'unknown')}%, "
    prompt += f"temperature {climate_conditions.get('temperature_mean', 'unknown')}°C, "
    prompt += f"and rainfall {climate_conditions.get('rainfall_mean', 'unknown')}mm, "
    prompt += f"recommend suitable crops: {', '.join([crop['crop'] for crop in suitable_crops[:3]])}"
    
    # Generate response
    response = self.finetuned_llm.generate_response(prompt, max_length=150, temperature=0.6)
    
    return response
'''
    
    return integration_code

if __name__ == "__main__":
    if integrate_finetuned_model():
        print("\n📝 Integration Code Generated:")
        print("=" * 60)
        print(create_model_integration_code())
        
        print("\n🎯 Next Steps:")
        print("1. Copy the integration code above")
        print("2. Add it to your main.py file")
        print("3. Modify the _generate_recommendation_text method")
        print("4. Test the web application")
        
        print("\n💡 Alternative: Use hybrid approach")
        print("- Fine-tuned model for agricultural queries")
        print("- Gemini API for general queries")
        print("- Fallback system for reliability")
