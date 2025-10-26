# Fallback Dataset Creation Script
# Use this if the main dataset file is missing

import os
import json
import pandas as pd

def create_fallback_dataset():
    """Create a dataset from available files if the main dataset is missing"""
    
    print("🔄 Creating fallback dataset from available files...")
    print("=" * 50)
    
    text_samples = []
    
    # Check for knowledge graph
    kg_path = '/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph.json'
    if os.path.exists(kg_path):
        print("✅ Loading knowledge graph...")
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        # Convert triples to text samples
        for i, triple in enumerate(kg_data[:1000]):  # Limit to 1000 for speed
            if isinstance(triple, dict) and 'subject' in triple and 'predicate' in triple and 'object' in triple:
                text = f"Agricultural fact: {triple['subject']} {triple['predicate']} {triple['object']}"
                text_samples.append({
                    'text': text,
                    'type': 'knowledge_fact',
                    'source': 'knowledge_graph'
                })
        print(f"  📝 Added {min(1000, len(kg_data))} knowledge facts")
    
    # Check for literature triples
    lit_path = '/content/drive/MyDrive/Final/data/processed/literature_triples.json'
    if os.path.exists(lit_path):
        print("✅ Loading literature triples...")
        with open(lit_path, 'r', encoding='utf-8') as f:
            lit_data = json.load(f)
        
        # Convert to text samples
        for i, triple in enumerate(lit_data[:500]):  # Limit to 500 for speed
            if isinstance(triple, dict) and 'subject' in triple and 'predicate' in triple and 'object' in triple:
                text = f"Research finding: {triple['subject']} {triple['predicate']} {triple['object']}"
                text_samples.append({
                    'text': text,
                    'type': 'research_finding',
                    'source': 'literature'
                })
        print(f"  📝 Added {min(500, len(lit_data))} research findings")
    
    # Check for Ugandan dataset
    csv_path = '/content/drive/MyDrive/Final/data/processed/ugandan_data_cleaned.csv'
    if os.path.exists(csv_path):
        print("✅ Loading Ugandan dataset...")
        df = pd.read_csv(csv_path)
        
        # Create text samples from crop data
        for i, row in df.head(200).iterrows():  # Limit to 200 for speed
            if 'crop' in row and 'soil_ph' in row and 'temperature_mean' in row:
                text = f"Crop recommendation: {row['crop']} grows well in soil pH {row['soil_ph']} and temperature {row['temperature_mean']}°C"
                text_samples.append({
                    'text': text,
                    'type': 'crop_recommendation',
                    'source': 'ugandan_data'
                })
        print(f"  📝 Added {min(200, len(df))} crop recommendations")
    
    # Add some general agricultural knowledge
    general_knowledge = [
        "Maize requires well-drained soil with pH between 5.5 and 7.5",
        "Rice grows best in clay soils with good water retention",
        "Beans are nitrogen-fixing crops that improve soil fertility",
        "Cassava is drought-tolerant and grows in poor soils",
        "Sweet potato prefers sandy loam soils with good drainage",
        "Coffee needs acidic soil with pH between 5.5 and 6.5",
        "Cotton requires warm temperatures and adequate rainfall",
        "Sugarcane needs fertile soil with good irrigation"
    ]
    
    for knowledge in general_knowledge:
        text_samples.append({
            'text': knowledge,
            'type': 'general_knowledge',
            'source': 'manual'
        })
    
    print(f"  📝 Added {len(general_knowledge)} general knowledge items")
    
    # Save the dataset
    output_path = '/content/drive/MyDrive/Final/data/processed/fallback_text_dataset.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(text_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Fallback dataset created!")
    print(f"📊 Total samples: {len(text_samples)}")
    print(f"💾 Saved to: {output_path}")
    
    # Show sample distribution
    type_counts = {}
    for sample in text_samples:
        sample_type = sample.get('type', 'unknown')
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
    
    print(f"\n📈 Sample distribution:")
    for sample_type, count in type_counts.items():
        print(f"  - {sample_type}: {count}")
    
    return output_path

if __name__ == "__main__":
    create_fallback_dataset()
