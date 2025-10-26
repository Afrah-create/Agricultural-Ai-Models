"""
Phase 4, Cell 3: Dataset to Knowledge Graph Triples Conversion
This cell converts the cleaned Ugandan dataset into knowledge graph triples
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset
print("Loading cleaned Ugandan dataset...")
df = pd.read_csv('/content/drive/MyDrive/Final/data/processed/ugandan_data_cleaned.csv')

print(f"Cleaned dataset loaded!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

def convert_dataset_to_triples(df):
    """
    Convert the cleaned Ugandan dataset into knowledge graph triples
    """
    
    triples = []
    BASE_URI = "http://example.org/agrokg/"
    
    # Helper function to create URIs
    def create_uri(entity_type, entity_name):
        sanitized_name = str(entity_name).replace(" ", "_").replace("/", "_").replace("-", "_").lower()
        return f"{BASE_URI}{entity_type}/{sanitized_name}"
    
    print("Converting dataset to knowledge graph triples...")
    
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"  Processing row {index+1}/{len(df)}")
        
        crop_name = row['crop_name_clean']
        crop_uri = create_uri("crop", crop_name)
        
        # 1. Crop - has_nutrient_requirement -> Nutrient Level
        nutrients = ['nitrogen', 'phosphorus', 'potassium']
        for nutrient in nutrients:
            if pd.notna(row[nutrient]):
                nutrient_uri = create_uri(f"{nutrient}_level", f"{row[nutrient]}_kg_per_ha")
                triples.append({
                    "subject": crop_uri,
                    "predicate": f"has_{nutrient}_requirement",
                    "object": nutrient_uri,
                    "value": float(row[nutrient]),
                    "unit": "kg/ha",
                    "source": "Ugandan_data_cleaned.csv",
                    "triple_type": "nutrient_requirement"
                })
        
        # 2. Crop - grows_in -> Agro-ecological Zone
        if pd.notna(row['agro_ecological_zone']):
            ae_zone = f"AEZ_{int(row['agro_ecological_zone'])}"
            ae_zone_uri = create_uri("agro_ecological_zone", ae_zone)
            triples.append({
                "subject": crop_uri,
                "predicate": "grows_in_agro_ecological_zone",
                "object": ae_zone_uri,
                "zone_id": int(row['agro_ecological_zone']),
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "zone_association"
            })
        
        # 3. Crop - requires_soil_property -> Soil Property Level
        soil_properties = ['pH', 'organic_matter']
        for prop in soil_properties:
            if pd.notna(row[prop]):
                prop_uri = create_uri(f"{prop.lower()}_level", f"{prop.lower()}_{row[prop]}")
                triples.append({
                    "subject": crop_uri,
                    "predicate": f"requires_{prop.lower()}",
                    "object": prop_uri,
                    "value": float(row[prop]),
                    "unit": "pH_scale" if prop == 'pH' else "percent",
                    "source": "Ugandan_data_cleaned.csv",
                    "triple_type": "soil_requirement"
                })
        
        # 4. Crop - prefers_texture -> Texture Class
        if pd.notna(row['texture_class']):
            texture_uri = create_uri("texture_class", row['texture_class'])
            triples.append({
                "subject": crop_uri,
                "predicate": "prefers_texture_class",
                "object": texture_uri,
                "texture": row['texture_class'],
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "texture_preference"
            })
        
        # 5. Crop - has_suitability_score -> Suitability Score
        if pd.notna(row['suitability_score']):
            score_uri = create_uri("suitability_score", f"score_{row['suitability_score']}")
            triples.append({
                "subject": crop_uri,
                "predicate": "has_suitability_score",
                "object": score_uri,
                "value": float(row['suitability_score']),
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "suitability_score"
            })
        
        # 6. Crop - has_yield_potential -> Yield Score
        if pd.notna(row['yield_score']):
            yield_uri = create_uri("yield_score", f"yield_{row['yield_score']}")
            triples.append({
                "subject": crop_uri,
                "predicate": "has_yield_potential",
                "object": yield_uri,
                "value": float(row['yield_score']),
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "yield_potential"
            })
        
        # 7. Crop - has_soil_quality_index -> Soil Quality Index
        if pd.notna(row['soil_quality_index']):
            soil_quality_uri = create_uri("soil_quality_index", f"quality_{row['soil_quality_index']}")
            triples.append({
                "subject": crop_uri,
                "predicate": "has_soil_quality_index",
                "object": soil_quality_uri,
                "value": float(row['soil_quality_index']),
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "soil_quality"
            })
        
        # 8. Crop - has_climate_suitability -> Climate Suitability
        if pd.notna(row['climate_suitability']):
            climate_uri = create_uri("climate_suitability", f"climate_{row['climate_suitability']}")
            triples.append({
                "subject": crop_uri,
                "predicate": "has_climate_suitability",
                "object": climate_uri,
                "value": float(row['climate_suitability']),
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "climate_suitability"
            })
        
        # 9. Crop - has_overall_suitability -> Overall Suitability
        if pd.notna(row['overall_suitability']):
            overall_uri = create_uri("overall_suitability", f"overall_{row['overall_suitability']}")
            triples.append({
                "subject": crop_uri,
                "predicate": "has_overall_suitability",
                "object": overall_uri,
                "value": float(row['overall_suitability']),
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "overall_suitability"
            })
        
        # 10. Crop - has_data_completeness -> Data Completeness
        if pd.notna(row['data_completeness']):
            completeness_uri = create_uri("data_completeness", f"completeness_{row['data_completeness']}")
            triples.append({
                "subject": crop_uri,
                "predicate": "has_data_completeness",
                "object": completeness_uri,
                "value": float(row['data_completeness']),
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "data_quality"
            })
        
        # 11. Crop - has_area_hectares -> Area Coverage
        if pd.notna(row['area_hectares']):
            area_uri = create_uri("area_coverage", f"area_{row['area_hectares']}")
            triples.append({
                "subject": crop_uri,
                "predicate": "has_area_coverage",
                "object": area_uri,
                "value": float(row['area_hectares']),
                "unit": "hectares",
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "area_coverage"
            })
        
        # 12. Crop - has_total_suitable_area -> Total Suitable Area
        if pd.notna(row['total_suitable_area']):
            total_area_uri = create_uri("total_suitable_area", f"total_{row['total_suitable_area']}")
            triples.append({
                "subject": crop_uri,
                "predicate": "has_total_suitable_area",
                "object": total_area_uri,
                "value": float(row['total_suitable_area']),
                "unit": "hectares",
                "source": "Ugandan_data_cleaned.csv",
                "triple_type": "total_suitable_area"
            })
    
    return triples

def analyze_dataset_triples(triples):
    """
    Analyze the dataset triples for quality and coverage
    """
    
    print(f"\nDataset Triple Analysis:")
    print(f"=" * 40)
    
    total_triples = len(triples)
    print(f"Total triples generated: {total_triples:,}")
    
    # Count by triple type
    triple_types = {}
    predicates = {}
    subjects = set()
    objects = set()
    
    for triple in triples:
        triple_type = triple.get('triple_type', 'unknown')
        triple_types[triple_type] = triple_types.get(triple_type, 0) + 1
        
        predicate = triple.get('predicate', 'unknown')
        predicates[predicate] = predicates.get(predicate, 0) + 1
        
        subjects.add(triple.get('subject', ''))
        objects.add(triple.get('object', ''))
    
    print(f"\nTriple Types:")
    for triple_type, count in sorted(triple_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {triple_type}: {count:,}")
    
    print(f"\nTop Predicates:")
    for predicate, count in sorted(predicates.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {predicate}: {count:,}")
    
    print(f"\nCoverage Analysis:")
    print(f"  Unique subjects: {len(subjects)}")
    print(f"  Unique objects: {len(objects)}")
    
    # Count unique crops
    crop_subjects = [s for s in subjects if 'crop/' in s]
    print(f"  Unique crops: {len(crop_subjects)}")
    
    return {
        'total_triples': total_triples,
        'triple_types': triple_types,
        'predicates': predicates,
        'unique_subjects': len(subjects),
        'unique_objects': len(objects),
        'unique_crops': len(crop_subjects)
    }

# Convert dataset to triples
dataset_triples = convert_dataset_to_triples(df)

# Analyze the triples
analysis = analyze_dataset_triples(dataset_triples)

# Save dataset triples
output_path = '/content/drive/MyDrive/Final/data/processed/dataset_triples.json'
with open(output_path, 'w') as f:
    json.dump(dataset_triples, f, indent=2)

print(f"\nDataset triples saved to: {output_path}")

# Save analysis
analysis_path = '/content/drive/MyDrive/Final/data/processed/dataset_triples_analysis.json'
with open(analysis_path, 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"Dataset triples analysis saved to: {analysis_path}")

# Display sample triples
print(f"\nSample Dataset Triples:")
print("-" * 30)

for i, triple in enumerate(dataset_triples[:5]):
    print(f"\nTriple {i+1}:")
    print(f"  Subject: {triple['subject']}")
    print(f"  Predicate: {triple['predicate']}")
    print(f"  Object: {triple['object']}")
    if 'value' in triple:
        print(f"  Value: {triple['value']}")
    if 'unit' in triple:
        print(f"  Unit: {triple['unit']}")
    print(f"  Type: {triple['triple_type']}")

print(f"\nDataset to Triples Conversion Complete!")
print(f"Generated {len(dataset_triples):,} triples from {len(df):,} records")
print(f"Next steps:")
print(f"  1. Integrate PDF and dataset triples")
print(f"  2. Build unified knowledge graph")
print(f"  3. Train graph embeddings")
