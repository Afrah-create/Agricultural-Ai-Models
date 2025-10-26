"""
Phase 3, Cell 1: Dataset Preprocessing and Analysis (Google Colab Version)
This cell preprocesses the Ugandan dataset and prepares it for knowledge graph construction
"""

# First, ensure we have the necessary imports and setup
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive (if not already mounted)
from google.colab import drive
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except:
    print("Google Drive already mounted or mounting failed")

# Note: Project structure already created from cell 03
print("Using existing project structure from cell 03...")

# Load the dataset
print("Loading Ugandan dataset...")
df = pd.read_csv('/content/drive/MyDrive/Final/Dataset/Ugandan_data.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Basic dataset information
print(f"\nDataset Overview:")
print(f"Total records: {len(df):,}")
print(f"Unique crops: {df['crop_name'].nunique()}")
print(f"Unique agro-ecological zones: {df['agro_ecological_zone'].nunique()}")
print(f"Date range: {df['processed_date'].min()} to {df['processed_date'].max()}")

# Check for missing values
print(f"\nMissing Values Analysis:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
else:
    print("No missing values found!")

# Crop analysis
print(f"\nCrop Analysis:")
crop_counts = df['crop_name'].value_counts()
print(f"Most common crops:")
print(crop_counts.head(10))

# Agro-ecological zone analysis
print(f"\nAgro-ecological Zone Analysis:")
zone_counts = df['agro_ecological_zone'].value_counts().sort_index()
print(f"Zone distribution:")
print(zone_counts)

# Soil properties analysis
print(f"\nSoil Properties Analysis:")
soil_props = ['pH', 'organic_matter', 'nitrogen', 'phosphorus', 'potassium']
for prop in soil_props:
    if prop in df.columns:
        print(f"{prop}: min={df[prop].min():.2f}, max={df[prop].max():.2f}, mean={df[prop].mean():.2f}")

# Suitability analysis
print(f"\nSuitability Analysis:")
suitability_cols = [col for col in df.columns if col.startswith('suitability_')]
for col in suitability_cols:
    total_area = df[col].sum()
    print(f"{col}: {total_area:,.2f} hectares")

# Data cleaning and preprocessing
print(f"\nData Cleaning and Preprocessing...")

# 1. Normalize crop names
df['crop_name_clean'] = df['crop_name'].str.lower().str.strip()

# Handle duplicate crop names with different cases
crop_mapping = {
    'maize': 'maize',
    'Maize': 'maize',
    'bean': 'beans',
    'Bean': 'beans',
    'beans': 'beans',
    'pea': 'peas',
    'Pea': 'peas',
    'peas': 'peas',
    'groundnut': 'groundnut',
    'groundnuts': 'groundnut',
    'cowpea': 'cowpea',
    'cowpeas': 'cowpea',
    'soyabeans': 'soybean',
    'sweet_potato': 'sweet_potato',
    'red pepper': 'red_pepper'
}

df['crop_name_clean'] = df['crop_name_clean'].map(crop_mapping).fillna(df['crop_name_clean'])

# 2. Create agro-ecological zone labels
df['zone_label'] = 'zone_' + df['agro_ecological_zone'].astype(str)

# 3. Calculate total suitability area
suitability_cols = [col for col in df.columns if col.startswith('suitability_') and col != 'suitability_not_suitable']
df['total_suitable_area_calc'] = df[suitability_cols].sum(axis=1)

# 4. Create suitability score (weighted average)
weights = {
    'suitability_very_suitable': 1.0,
    'suitability_suitable': 0.8,
    'suitability_moderately_suitable': 0.6,
    'suitability_marginally_suitable': 0.4,
    'suitability_very_marginally_suitable': 0.2
}

df['suitability_score'] = 0
for col, weight in weights.items():
    if col in df.columns:
        df['suitability_score'] += df[col] * weight

# Normalize by total area
df['suitability_score'] = df['suitability_score'] / df['area_hectares']
df['suitability_score'] = df['suitability_score'].fillna(0)

# 5. Create yield score (weighted average)
yield_cols = [col for col in df.columns if col.startswith('yield_') and col != 'yield_maximum']
yield_weights = {
    'yield_very_suitable': 1.0,
    'yield_suitable': 0.8,
    'yield_moderately_suitable': 0.6,
    'yield_marginally_suitable': 0.4,
    'yield_very_marginally_suitable': 0.2
}

df['yield_score'] = 0
for col, weight in yield_weights.items():
    if col in df.columns:
        df['yield_score'] += df[col] * weight

# Normalize by maximum yield
df['yield_score'] = df['yield_score'] / df['yield_maximum']
df['yield_score'] = df['yield_score'].fillna(0)

# 6. Create soil quality index
soil_quality_factors = []
if 'pH' in df.columns:
    # Optimal pH range 6.0-7.0
    ph_score = np.where((df['pH'] >= 6.0) & (df['pH'] <= 7.0), 1.0,
                       np.where((df['pH'] >= 5.5) & (df['pH'] <= 7.5), 0.7, 0.3))
    soil_quality_factors.append(ph_score)

if 'organic_matter' in df.columns:
    # Higher organic matter is better
    om_score = np.where(df['organic_matter'] >= 3, 1.0,
                       np.where(df['organic_matter'] >= 2, 0.7, 0.4))
    soil_quality_factors.append(om_score)

if soil_quality_factors:
    df['soil_quality_index'] = np.mean(soil_quality_factors, axis=0)
else:
    df['soil_quality_index'] = 0.5

# 7. Create climate suitability
climate_factors = []
if 'temperature_mean' in df.columns:
    # Optimal temperature range 20-30°C for most crops
    temp_score = np.where((df['temperature_mean'] >= 20) & (df['temperature_mean'] <= 30), 1.0,
                         np.where((df['temperature_mean'] >= 15) & (df['temperature_mean'] <= 35), 0.7, 0.4))
    climate_factors.append(temp_score)

if 'rainfall_mean' in df.columns:
    # Optimal rainfall range 800-1500 mm
    rain_score = np.where((df['rainfall_mean'] >= 800) & (df['rainfall_mean'] <= 1500), 1.0,
                         np.where((df['rainfall_mean'] >= 600) & (df['rainfall_mean'] <= 2000), 0.7, 0.4))
    climate_factors.append(rain_score)

if climate_factors:
    df['climate_suitability'] = np.mean(climate_factors, axis=0)
else:
    df['climate_suitability'] = 0.5

# 8. Create overall suitability index
df['overall_suitability'] = (df['suitability_score'] * 0.4 + 
                            df['yield_score'] * 0.3 + 
                            df['soil_quality_index'] * 0.2 + 
                            df['climate_suitability'] * 0.1)

# Data validation
print(f"\nData Validation:")
print(f"Records with valid suitability scores: {(df['suitability_score'] > 0).sum()}")
print(f"Records with valid yield scores: {(df['yield_score'] > 0).sum()}")
print(f"Records with valid overall suitability: {(df['overall_suitability'] > 0).sum()}")

# Save preprocessed data
output_path = '/content/drive/MyDrive/Final/data/processed/ugandan_data_preprocessed.csv'
df.to_csv(output_path, index=False)
print(f"\nPreprocessed data saved to: {output_path}")

# Create summary statistics
summary_stats = {
    'total_records': len(df),
    'unique_crops': df['crop_name_clean'].nunique(),
    'unique_zones': df['agro_ecological_zone'].nunique(),
    'crop_distribution': df['crop_name_clean'].value_counts().to_dict(),
    'zone_distribution': df['agro_ecological_zone'].value_counts().to_dict(),
    'soil_properties': {
        'pH': {'min': float(df['pH'].min()), 'max': float(df['pH'].max()), 'mean': float(df['pH'].mean())},
        'organic_matter': {'min': float(df['organic_matter'].min()), 'max': float(df['organic_matter'].max()), 'mean': float(df['organic_matter'].mean())},
        'nitrogen': {'min': float(df['nitrogen'].min()), 'max': float(df['nitrogen'].max()), 'mean': float(df['nitrogen'].mean())},
        'phosphorus': {'min': float(df['phosphorus'].min()), 'max': float(df['phosphorus'].max()), 'mean': float(df['phosphorus'].mean())},
        'potassium': {'min': float(df['potassium'].min()), 'max': float(df['potassium'].max()), 'mean': float(df['potassium'].mean())}
    },
    'suitability_stats': {
        'avg_suitability_score': float(df['suitability_score'].mean()),
        'avg_yield_score': float(df['yield_score'].mean()),
        'avg_soil_quality': float(df['soil_quality_index'].mean()),
        'avg_climate_suitability': float(df['climate_suitability'].mean()),
        'avg_overall_suitability': float(df['overall_suitability'].mean())
    },
    'preprocessing_timestamp': datetime.now().isoformat()
}

# Save summary statistics
summary_path = '/content/drive/MyDrive/Final/data/processed/dataset_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"Summary statistics saved to: {summary_path}")

# Display final summary
print(f"\nPreprocessing Complete!")
print(f"=" * 50)
print(f"Total records processed: {len(df):,}")
print(f"Unique crops: {df['crop_name_clean'].nunique()}")
print(f"Unique zones: {df['agro_ecological_zone'].nunique()}")
print(f"Average suitability score: {df['suitability_score'].mean():.3f}")
print(f"Average yield score: {df['yield_score'].mean():.3f}")
print(f"Average overall suitability: {df['overall_suitability'].mean():.3f}")

print(f"\nTop 10 crops by record count:")
top_crops = df['crop_name_clean'].value_counts().head(10)
for crop, count in top_crops.items():
    print(f"  {crop}: {count:,} records")

print(f"\nNext steps:")
print(f"  1. Convert preprocessed data to knowledge graph triples")
print(f"  2. Integrate with PDF-extracted triples")
print(f"  3. Build unified knowledge graph")
print(f"  4. Train graph embeddings")

# Display sample of preprocessed data
print(f"\nSample of preprocessed data:")
print(df[['crop_name_clean', 'zone_label', 'suitability_score', 'yield_score', 'soil_quality_index', 'overall_suitability']].head())
