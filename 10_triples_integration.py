"""
Phase 4, Cell 5: PDF and Dataset Triples Integration
This cell integrates PDF-extracted triples with dataset triples to create a unified knowledge graph
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Loading Knowledge Graph Components...")

# Load all components
try:
    # Load dataset triples
    with open('/content/drive/MyDrive/Final/data/processed/dataset_triples.json', 'r') as f:
        dataset_triples = json.load(f)
    print(f"✅ Dataset triples loaded: {len(dataset_triples):,}")
    
    # Load literature triples
    with open('/content/drive/MyDrive/Final/data/processed/complete_literature_triples.json', 'r') as f:
        literature_triples = json.load(f)
    print(f"✅ Literature triples loaded: {len(literature_triples)} PDFs")
    
    # Load analysis reports
    with open('/content/drive/MyDrive/Final/data/processed/dataset_triples_analysis.json', 'r') as f:
        dataset_analysis = json.load(f)
    
    with open('/content/drive/MyDrive/Final/data/processed/complete_literature_analysis.json', 'r') as f:
        literature_analysis = json.load(f)
    
    print(f"✅ Analysis reports loaded")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("Please ensure all previous cells have been run successfully")

def extract_literature_relationships(literature_triples):
    """
    Extract structured relationships from literature triples
    """
    
    print("Extracting literature relationships...")
    
    extracted_relationships = []
    BASE_URI = "http://example.org/agrokg/"
    
    def create_uri(entity_type, entity_name):
        sanitized_name = str(entity_name).replace(" ", "_").replace("/", "_").replace("-", "_").lower()
        return f"{BASE_URI}{entity_type}/{sanitized_name}"
    
    for pdf_data in literature_triples:
        pdf_name = pdf_data.get('pdf_file', 'unknown')
        
        # Extract soil-crop relationships
        if 'soil_crop_relationships' in pdf_data and isinstance(pdf_data['soil_crop_relationships'], list):
            for rel in pdf_data['soil_crop_relationships']:
                if isinstance(rel, dict) and 'subject' in rel and 'predicate' in rel and 'object' in rel:
                    relationship = {
                        "subject": create_uri("soil_property", rel['subject']),
                        "predicate": rel['predicate'],
                        "object": create_uri("crop", rel['object']),
                        "conditions": rel.get('conditions', ''),
                        "threshold": rel.get('threshold', ''),
                        "evidence": rel.get('evidence', ''),
                        "confidence": rel.get('confidence', 0.5),
                        "source": f"literature_{pdf_name}",
                        "triple_type": "literature_soil_crop_relationship"
                    }
                    extracted_relationships.append(relationship)
        
        # Extract management practices
        if 'management_practices' in pdf_data and isinstance(pdf_data['management_practices'], list):
            for practice in pdf_data['management_practices']:
                if isinstance(practice, dict) and 'practice_type' in practice and 'crop' in practice:
                    relationship = {
                        "subject": create_uri("management_practice", practice['practice_type']),
                        "predicate": "recommended_for",
                        "object": create_uri("crop", practice['crop']),
                        "soil_condition": practice.get('soil_condition', ''),
                        "application_rate": practice.get('application_rate', ''),
                        "timing": practice.get('timing', ''),
                        "effect": practice.get('effect', ''),
                        "evidence": practice.get('evidence', ''),
                        "source": f"literature_{pdf_name}",
                        "triple_type": "literature_management_practice"
                    }
                    extracted_relationships.append(relationship)
        
        # Extract climate requirements
        if 'climate_requirements' in pdf_data and isinstance(pdf_data['climate_requirements'], list):
            for climate in pdf_data['climate_requirements']:
                if isinstance(climate, dict) and 'crop' in climate:
                    relationship = {
                        "subject": create_uri("crop", climate['crop']),
                        "predicate": "requires_climate",
                        "object": create_uri("climate_zone", "general"),
                        "temperature_range": climate.get('temperature_range', ''),
                        "rainfall_requirement": climate.get('rainfall_requirement', ''),
                        "growing_season": climate.get('growing_season', ''),
                        "evidence": climate.get('evidence', ''),
                        "source": f"literature_{pdf_name}",
                        "triple_type": "literature_climate_requirement"
                    }
                    extracted_relationships.append(relationship)
    
    print(f"Extracted {len(extracted_relationships)} literature relationships")
    return extracted_relationships

def normalize_entity_names(triples):
    """
    Normalize entity names across dataset and literature triples
    """
    
    print("Normalizing entity names...")
    
    # Common crop name mappings
    crop_mappings = {
        'maize': 'maize',
        'Maize': 'maize',
        'Maize (Zea mays L.)': 'maize',
        'Maize yield': 'maize',
        'Maize yields': 'maize',
        'corn': 'maize',
        'rice': 'rice',
        'lowland rice': 'rice',
        'upland rice': 'rice',
        'beans': 'beans',
        'Common bean (Phaseolus vulgaris L.)': 'beans',
        'bean': 'beans',
        'cassava': 'cassava',
        'banana': 'banana',
        'Banana': 'banana',
        'FHIA17, M9, M2 banana varieties': 'banana',
        'coffee': 'coffee',
        'Coffee (Arabica and Robusta)': 'coffee',
        'Coffee (general)': 'coffee',
        'groundnut': 'groundnut',
        'groundnuts': 'groundnut',
        'peanut': 'groundnut',
        'cotton': 'cotton',
        'sugarcane': 'sugarcane',
        'sweet_potato': 'sweet_potato',
        'yam': 'yam',
        'wheat': 'wheat',
        'Wheat': 'wheat',
        'millet': 'millet',
        'finger millet': 'millet',
        'foxtail millet': 'millet',
        'pearl millet': 'millet',
        'sunflower': 'sunflower',
        'tea': 'tea',
        'soybean': 'soybean',
        'soyabeans': 'soybean',
        'cowpea': 'cowpea',
        'chickpea': 'chickpea',
        'pea': 'peas',
        'peas': 'peas'
    }
    
    # Common soil property mappings
    soil_mappings = {
        'pH': 'ph',
        'soil pH': 'ph',
        'pH level': 'ph',
        'organic matter': 'organic_matter',
        'Organic Matter (OM)': 'organic_matter',
        'soil organic matter content': 'organic_matter',
        'nitrogen': 'nitrogen',
        'Nitrogen': 'nitrogen',
        'N': 'nitrogen',
        'phosphorus': 'phosphorus',
        'Phosphorus': 'phosphorus',
        'P': 'phosphorus',
        'potassium': 'potassium',
        'Potassium': 'potassium',
        'K': 'potassium',
        'Potassium (K)': 'potassium',
        'CEC': 'cec',
        'cation exchange capacity': 'cec',
        'texture': 'texture',
        'soil texture': 'texture',
        'texture class': 'texture'
    }
    
    normalized_triples = []
    
    for triple in triples:
        normalized_triple = triple.copy()
        
        # Normalize crop names in subjects and objects
        if 'crop/' in triple.get('subject', ''):
            crop_name = triple['subject'].split('/')[-1]
            if crop_name in crop_mappings:
                normalized_triple['subject'] = triple['subject'].replace(crop_name, crop_mappings[crop_name])
        
        if 'crop/' in triple.get('object', ''):
            crop_name = triple['object'].split('/')[-1]
            if crop_name in crop_mappings:
                normalized_triple['object'] = triple['object'].replace(crop_name, crop_mappings[crop_name])
        
        # Normalize soil property names
        if 'soil_property/' in triple.get('subject', ''):
            soil_name = triple['subject'].split('/')[-1]
            if soil_name in soil_mappings:
                normalized_triple['subject'] = triple['subject'].replace(soil_name, soil_mappings[soil_name])
        
        if 'soil_property/' in triple.get('object', ''):
            soil_name = triple['object'].split('/')[-1]
            if soil_name in soil_mappings:
                normalized_triple['object'] = triple['object'].replace(soil_name, soil_mappings[soil_name])
        
        normalized_triples.append(normalized_triple)
    
    print(f"Normalized {len(normalized_triples)} triples")
    return normalized_triples

def integrate_triples(dataset_triples, literature_relationships):
    """
    Integrate dataset and literature triples into a unified knowledge graph
    """
    
    print("Integrating triples into unified knowledge graph...")
    
    # Normalize both sets of triples
    normalized_dataset_triples = normalize_entity_names(dataset_triples)
    normalized_literature_triples = normalize_entity_names(literature_relationships)
    
    # Combine all triples
    unified_triples = normalized_dataset_triples + normalized_literature_triples
    
    # Add integration metadata
    for i, triple in enumerate(unified_triples):
        triple['triple_id'] = f"triple_{i+1:06d}"
        triple['integration_timestamp'] = datetime.now().isoformat()
    
    print(f"Unified knowledge graph created with {len(unified_triples):,} triples")
    
    return unified_triples

def analyze_unified_knowledge_graph(unified_triples):
    """
    Analyze the unified knowledge graph
    """
    
    print(f"\nUnified Knowledge Graph Analysis:")
    print(f"=" * 50)
    
    total_triples = len(unified_triples)
    
    # Count by source
    dataset_count = sum(1 for t in unified_triples if 'Ugandan_data_cleaned.csv' in t.get('source', ''))
    literature_count = sum(1 for t in unified_triples if 'literature_' in t.get('source', ''))
    
    # Count by triple type
    triple_types = {}
    predicates = {}
    subjects = set()
    objects = set()
    
    for triple in unified_triples:
        triple_type = triple.get('triple_type', 'unknown')
        triple_types[triple_type] = triple_types.get(triple_type, 0) + 1
        
        predicate = triple.get('predicate', 'unknown')
        predicates[predicate] = predicates.get(predicate, 0) + 1
        
        subjects.add(triple.get('subject', ''))
        objects.add(triple.get('object', ''))
    
    print(f"Total triples: {total_triples:,}")
    print(f"Dataset triples: {dataset_count:,} ({dataset_count/total_triples*100:.1f}%)")
    print(f"Literature triples: {literature_count:,} ({literature_count/total_triples*100:.1f}%)")
    
    print(f"\nTriple Types:")
    for triple_type, count in sorted(triple_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {triple_type}: {count:,}")
    
    print(f"\nTop Predicates:")
    for predicate, count in sorted(predicates.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {predicate}: {count:,}")
    
    print(f"\nCoverage Analysis:")
    print(f"  Unique subjects: {len(subjects)}")
    print(f"  Unique objects: {len(objects)}")
    
    # Count unique crops
    crop_subjects = [s for s in subjects if 'crop/' in s]
    crop_objects = [o for o in objects if 'crop/' in o]
    unique_crops = set(crop_subjects + crop_objects)
    print(f"  Unique crops: {len(unique_crops)}")
    
    # Count unique soil properties
    soil_subjects = [s for s in subjects if 'soil_property/' in s]
    soil_objects = [o for o in objects if 'soil_property/' in o]
    unique_soil_props = set(soil_subjects + soil_objects)
    print(f"  Unique soil properties: {len(unique_soil_props)}")
    
    return {
        'total_triples': total_triples,
        'dataset_triples': dataset_count,
        'literature_triples': literature_count,
        'triple_types': triple_types,
        'predicates': predicates,
        'unique_subjects': len(subjects),
        'unique_objects': len(objects),
        'unique_crops': len(unique_crops),
        'unique_soil_properties': len(unique_soil_props)
    }

def create_knowledge_graph_schema(unified_triples):
    """
    Create a comprehensive schema for the unified knowledge graph
    """
    
    print("Creating unified knowledge graph schema...")
    
    schema = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_triples': len(unified_triples),
            'version': '1.0.0',
            'description': 'Unified Agricultural Knowledge Graph for Uganda'
        },
        'node_types': {
            'crop': {
                'description': 'Agricultural crops and varieties',
                'attributes': ['name', 'scientific_name', 'family', 'growth_period'],
                'examples': ['maize', 'rice', 'beans', 'cassava', 'banana']
            },
            'soil_property': {
                'description': 'Soil properties and characteristics',
                'attributes': ['name', 'unit', 'min_value', 'max_value', 'optimal_range'],
                'examples': ['ph', 'organic_matter', 'nitrogen', 'phosphorus', 'potassium']
            },
            'agro_ecological_zone': {
                'description': 'Agro-ecological zones in Uganda',
                'attributes': ['zone_id', 'description', 'climate_characteristics'],
                'examples': ['AEZ_1', 'AEZ_2', 'AEZ_3', 'AEZ_5', 'AEZ_6']
            },
            'management_practice': {
                'description': 'Agricultural management practices',
                'attributes': ['type', 'description', 'application_rate', 'timing'],
                'examples': ['fertilization', 'liming', 'irrigation', 'mulching']
            },
            'climate_zone': {
                'description': 'Climate zones and requirements',
                'attributes': ['name', 'temperature_range', 'rainfall_range'],
                'examples': ['tropical_wet', 'tropical_dry', 'semi_arid']
            }
        },
        'relationship_types': {
            'has_nutrient_requirement': {
                'description': 'Crop requires specific nutrient level',
                'weight_range': [0, 1],
                'attributes': ['value', 'unit', 'source']
            },
            'grows_in_agro_ecological_zone': {
                'description': 'Crop grows in specific agro-ecological zone',
                'attributes': ['zone_id', 'suitability_score']
            },
            'requires_ph': {
                'description': 'Crop requires specific pH level',
                'attributes': ['value', 'unit', 'optimal_range']
            },
            'prefers_texture_class': {
                'description': 'Crop prefers specific soil texture',
                'attributes': ['texture_type', 'suitability']
            },
            'suitability': {
                'description': 'Soil-crop suitability relationship',
                'weight_range': [0, 1],
                'attributes': ['conditions', 'threshold', 'evidence']
            },
            'recommended_for': {
                'description': 'Management practice recommended for crop',
                'attributes': ['application_rate', 'timing', 'effect']
            },
            'requires_climate': {
                'description': 'Crop requires specific climate conditions',
                'attributes': ['temperature_range', 'rainfall_requirement']
            }
        },
        'data_sources': {
            'dataset': 'Ugandan agricultural dataset (quantitative)',
            'literature': 'Agricultural literature reviews (qualitative)'
        }
    }
    
    return schema

# Execute integration process
print("Starting PDF and Dataset Triples Integration...")

# Extract literature relationships
literature_relationships = extract_literature_relationships(literature_triples)

# Integrate triples
unified_triples = integrate_triples(dataset_triples, literature_relationships)

# Analyze unified knowledge graph
analysis = analyze_unified_knowledge_graph(unified_triples)

# Create schema
schema = create_knowledge_graph_schema(unified_triples)

# Save unified knowledge graph
output_path = '/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph.json'
with open(output_path, 'w') as f:
    json.dump(unified_triples, f, indent=2)

print(f"\nUnified knowledge graph saved to: {output_path}")

# Save analysis
analysis_path = '/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph_analysis.json'
with open(analysis_path, 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"Unified knowledge graph analysis saved to: {analysis_path}")

# Save schema
schema_path = '/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph_schema.json'
with open(schema_path, 'w') as f:
    json.dump(schema, f, indent=2)

print(f"Unified knowledge graph schema saved to: {schema_path}")

# Display sample integrated triples
print(f"\nSample Integrated Triples:")
print("-" * 30)

# Show dataset triples
print(f"\nDataset Triples (Quantitative):")
for i, triple in enumerate([t for t in unified_triples if 'Ugandan_data_cleaned.csv' in t.get('source', '')][:3]):
    print(f"\nTriple {i+1}:")
    print(f"  Subject: {triple['subject']}")
    print(f"  Predicate: {triple['predicate']}")
    print(f"  Object: {triple['object']}")
    if 'value' in triple:
        print(f"  Value: {triple['value']}")
    print(f"  Type: {triple['triple_type']}")

# Show literature triples
print(f"\nLiterature Triples (Qualitative):")
for i, triple in enumerate([t for t in unified_triples if 'literature_' in t.get('source', '')][:3]):
    print(f"\nTriple {i+1}:")
    print(f"  Subject: {triple['subject']}")
    print(f"  Predicate: {triple['predicate']}")
    print(f"  Object: {triple['object']}")
    if 'evidence' in triple:
        print(f"  Evidence: {triple['evidence'][:100]}...")
    print(f"  Type: {triple['triple_type']}")

print(f"\n🎉 PDF and Dataset Triples Integration Complete!")
print(f"=" * 60)
print(f"📊 Total triples: {analysis['total_triples']:,}")
print(f"📈 Dataset triples: {analysis['dataset_triples']:,} ({analysis['dataset_triples']/analysis['total_triples']*100:.1f}%)")
print(f"📚 Literature triples: {analysis['literature_triples']:,} ({analysis['literature_triples']/analysis['total_triples']*100:.1f}%)")
print(f"🌾 Unique crops: {analysis['unique_crops']}")
print(f"🌍 Unique soil properties: {analysis['unique_soil_properties']}")
print(f"🔗 Unique subjects: {analysis['unique_subjects']}")
print(f"🎯 Unique objects: {analysis['unique_objects']}")

print(f"\nNext steps:")
print(f"  1. Build graph structure with NetworkX")
print(f"  2. Train graph embeddings")
print(f"  3. Develop RAG pipeline")
print(f"  4. Create crop recommendation system")
