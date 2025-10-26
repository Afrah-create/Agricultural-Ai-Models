"""
Phase 7, Cell 2: Agricultural Text Dataset Creation
This cell creates a comprehensive text dataset from your knowledge graph and literature data
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import re
from collections import defaultdict

print("🔄 Creating Agricultural Text Dataset for LLM Fine-tuning...")

class AgriculturalTextDataset:
    """Create text dataset from agricultural knowledge graph and literature"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.text_samples = []
        
    def load_data_sources(self):
        """Load all available data sources"""
        print("Loading data sources...")
        
        # Load unified knowledge graph
        kg_path = os.path.join(self.data_dir, 'unified_knowledge_graph.json')
        if os.path.exists(kg_path):
            with open(kg_path, 'r', encoding='utf-8') as f:
                self.knowledge_graph = json.load(f)
            print(f"✅ Knowledge graph: {len(self.knowledge_graph):,} triples")
        else:
            self.knowledge_graph = []
            
        # Load literature triples
        lit_path = os.path.join(self.data_dir, 'complete_literature_triples.json')
        if os.path.exists(lit_path):
            with open(lit_path, 'r', encoding='utf-8') as f:
                self.literature_triples = json.load(f)
            print(f"✅ Literature triples: {len(self.literature_triples):,} triples")
        else:
            self.literature_triples = []
            
        # Load cleaned dataset
        csv_path = os.path.join(self.data_dir, 'ugandan_data_cleaned.csv')
        if os.path.exists(csv_path):
            self.ugandan_data = pd.read_csv(csv_path)
            print(f"✅ Ugandan dataset: {len(self.ugandan_data):,} records")
        else:
            self.ugandan_data = None
    
    def create_crop_recommendation_samples(self):
        """Create crop recommendation text samples"""
        print("Creating crop recommendation samples...")
        
        if self.ugandan_data is not None:
            for _, row in self.ugandan_data.iterrows():
                # Create recommendation text
                crop = row.get('crop_name_clean', 'Unknown')
                ph = row.get('pH', 'Unknown')
                temp = row.get('temperature_mean', 'Unknown')
                rainfall = row.get('rainfall_mean', 'Unknown')
                soil_texture = row.get('texture_class', 'Unknown')
                
                # Create natural language recommendation
                text = f"""Agricultural Recommendation:

Soil Conditions: pH {ph}, {soil_texture} soil texture
Climate Conditions: {temp}°C average temperature, {rainfall}mm annual rainfall

Recommendation: {crop.title()} is suitable for these conditions.

Reasoning: Based on agricultural research, {crop} thrives in soils with pH around {ph} and {soil_texture} texture. The temperature of {temp}°C and rainfall of {rainfall}mm provide optimal growing conditions for {crop} cultivation.

Management Practices:
- Ensure proper soil preparation with adequate drainage
- Monitor soil pH and adjust if necessary
- Implement appropriate irrigation based on rainfall patterns
- Use recommended fertilizers for {crop} cultivation
- Follow crop rotation practices to maintain soil health

Expected Yield: Under optimal conditions, {crop} can achieve good yields with proper management.

This recommendation is based on agricultural research and local conditions in Uganda."""
                
                self.text_samples.append({
                    'text': text,
                    'type': 'crop_recommendation',
                    'crop': crop,
                    'source': 'ugandan_dataset'
                })
    
    def create_knowledge_graph_samples(self):
        """Create text samples from knowledge graph triples"""
        print("Creating knowledge graph text samples...")
        
        # Group triples by subject for better context
        grouped_triples = defaultdict(list)
        for triple in self.knowledge_graph:
            subject = triple.get('subject', '')
            grouped_triples[subject].append(triple)
        
        for subject, triples in list(grouped_triples.items())[:1000]:  # Limit for performance
            if len(triples) < 2:  # Skip subjects with too few triples
                continue
                
            # Extract crop name from subject
            crop_name = subject.split('/')[-1].replace('_', ' ').title()
            
            # Create comprehensive text about the crop
            text_parts = [f"Agricultural Knowledge about {crop_name}:"]
            
            for triple in triples[:10]:  # Limit triples per subject
                predicate = triple.get('predicate', '')
                obj = triple.get('object', '')
                
                # Convert predicate to natural language
                if 'has_nutrient_requirement' in predicate:
                    text_parts.append(f"- Nutrient requirements: {obj}")
                elif 'suitable_for_soil' in predicate:
                    text_parts.append(f"- Suitable soil types: {obj}")
                elif 'requires_climate' in predicate:
                    text_parts.append(f"- Climate requirements: {obj}")
                elif 'has_growing_season' in predicate:
                    text_parts.append(f"- Growing season: {obj}")
                elif 'has_water_requirement' in predicate:
                    text_parts.append(f"- Water requirements: {obj}")
                else:
                    text_parts.append(f"- {predicate.replace('_', ' ').title()}: {obj}")
            
            if len(text_parts) > 1:
                text = "\n".join(text_parts)
                text += f"\n\nThis information is based on agricultural research and knowledge graphs."
                
                self.text_samples.append({
                    'text': text,
                    'type': 'agricultural_knowledge',
                    'crop': crop_name,
                    'source': 'knowledge_graph'
                })
    
    def create_literature_samples(self):
        """Create text samples from literature triples"""
        print("Creating literature text samples...")
        
        for triple in self.literature_triples[:500]:  # Limit for performance
            subject = triple.get('subject', '')
            predicate = triple.get('predicate', '')
            obj = triple.get('object', '')
            evidence = triple.get('evidence', '')
            
            if evidence and len(evidence) > 50:  # Only use substantial evidence
                text = f"""Research Finding:

Subject: {subject.replace('_', ' ').title()}
Finding: {predicate.replace('_', ' ').title()} - {obj}

Evidence: {evidence}

This finding is supported by agricultural research literature and provides valuable insights for crop cultivation practices."""
                
                self.text_samples.append({
                    'text': text,
                    'type': 'research_finding',
                    'crop': subject.split('/')[-1].replace('_', ' ').title(),
                    'source': 'literature'
                })
    
    def create_qa_samples(self):
        """Create question-answer pairs for training"""
        print("Creating Q&A samples...")
        
        qa_pairs = [
            {
                'question': "What crops are suitable for loamy soil with pH 6.5?",
                'answer': "Crops suitable for loamy soil with pH 6.5 include maize, beans, cassava, and sweet potato. Loamy soil provides good drainage and nutrient retention, while pH 6.5 is optimal for most crops. Maize performs particularly well in these conditions, achieving good yields with proper management."
            },
            {
                'question': "How much rainfall does rice require?",
                'answer': "Rice requires 1000-2000mm of annual rainfall for optimal growth. It thrives in wet conditions and can tolerate flooding. In areas with lower rainfall, supplementary irrigation is necessary. Rice cultivation is most successful in regions with consistent water availability throughout the growing season."
            },
            {
                'question': "What are the temperature requirements for cassava?",
                'answer': "Cassava grows best in temperatures between 20-35°C, with optimal range of 25-30°C. It is a tropical crop that requires warm conditions and cannot tolerate frost. Cassava is drought-tolerant and can survive in areas with limited rainfall, making it suitable for various climatic conditions."
            },
            {
                'question': "How do I improve soil pH for crop cultivation?",
                'answer': "To increase soil pH (make it less acidic), apply agricultural lime or dolomite. To decrease soil pH (make it more acidic), apply sulfur or organic matter. The amount needed depends on current pH, target pH, and soil type. Regular soil testing helps monitor pH changes and adjust management practices accordingly."
            },
            {
                'question': "What is crop rotation and why is it important?",
                'answer': "Crop rotation is the practice of growing different crops in the same area in sequential seasons. It helps maintain soil fertility, reduces pest and disease pressure, improves soil structure, and enhances nutrient cycling. Common rotation patterns include alternating legumes with cereals to fix nitrogen and break pest cycles."
            }
        ]
        
        for qa in qa_pairs:
            text = f"""Agricultural Question and Answer:

Question: {qa['question']}

Answer: {qa['answer']}

This information is based on agricultural research and best practices for sustainable farming."""
            
            self.text_samples.append({
                'text': text,
                'type': 'qa_pair',
                'crop': 'general',
                'source': 'expert_knowledge'
            })
    
    def generate_dataset(self):
        """Generate the complete text dataset"""
        print("🔄 Generating agricultural text dataset...")
        
        self.load_data_sources()
        self.create_crop_recommendation_samples()
        self.create_knowledge_graph_samples()
        self.create_literature_samples()
        self.create_qa_samples()
        
        print(f"\n✅ Dataset generation complete!")
        print(f"Total text samples: {len(self.text_samples):,}")
        
        # Show distribution by type
        type_counts = defaultdict(int)
        for sample in self.text_samples:
            type_counts[sample['type']] += 1
        
        print("\n📊 Sample distribution by type:")
        for sample_type, count in type_counts.items():
            print(f"  {sample_type}: {count:,} samples")
        
        return self.text_samples

# Create the dataset
dataset_creator = AgriculturalTextDataset(DATA_DIR)
text_samples = dataset_creator.generate_dataset()

# Save the dataset
dataset_path = os.path.join(DATA_DIR, 'agricultural_text_dataset.json')
with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(text_samples, f, indent=2, ensure_ascii=False)

print(f"\n💾 Dataset saved to: {dataset_path}")
print(f"File size: {os.path.getsize(dataset_path) / (1024*1024):.1f} MB")

# Show sample texts
print("\n📝 Sample texts:")
for i, sample in enumerate(text_samples[:3]):
    print(f"\n--- Sample {i+1} ({sample['type']}) ---")
    print(sample['text'][:300] + "..." if len(sample['text']) > 300 else sample['text'])
