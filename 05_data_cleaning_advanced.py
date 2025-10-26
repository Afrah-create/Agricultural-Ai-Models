"""
Phase 3, Cell 2: Advanced Data Cleaning and Null Value Handling
This cell performs advanced cleaning of the preprocessed dataset
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the preprocessed dataset
print("Loading preprocessed dataset...")
df = pd.read_csv('/content/drive/MyDrive/Final/data/processed/ugandan_data_preprocessed.csv')

print(f"Preprocessed dataset loaded!")
print(f"Shape: {df.shape}")

# Detailed missing value analysis
print(f"\nDetailed Missing Value Analysis:")
missing_analysis = df.isnull().sum()
missing_percentage = (missing_analysis / len(df)) * 100

missing_summary = pd.DataFrame({
    'Column': missing_analysis.index,
    'Missing_Count': missing_analysis.values,
    'Missing_Percentage': missing_percentage.values
}).sort_values('Missing_Percentage', ascending=False)

print(missing_summary[missing_summary['Missing_Count'] > 0])

# Strategy for handling missing values
print(f"\nData Cleaning Strategy:")

# 1. Handle climate data (86% missing)
print("1. Climate Data Handling:")
climate_cols = ['temperature_mean', 'humidity_mean', 'rainfall_mean', 'region']

# For climate data, we'll use zone-based imputation
zone_climate_stats = {}
for zone in df['agro_ecological_zone'].dropna().unique():
    zone_data = df[df['agro_ecological_zone'] == zone]
    zone_stats = {
        'temperature_mean': zone_data['temperature_mean'].mean(),
        'humidity_mean': zone_data['humidity_mean'].mean(),
        'rainfall_mean': zone_data['rainfall_mean'].mean(),
        'region': zone_data['region'].mode().iloc[0] if not zone_data['region'].mode().empty else 'Unknown'
    }
    zone_climate_stats[zone] = zone_stats

print(f"   Created climate statistics for {len(zone_climate_stats)} zones")

# Impute climate data based on zone
for idx, row in df.iterrows():
    if pd.isna(row['temperature_mean']) and pd.notna(row['agro_ecological_zone']):
        zone = row['agro_ecological_zone']
        if zone in zone_climate_stats:
            df.at[idx, 'temperature_mean'] = zone_climate_stats[zone]['temperature_mean']
            df.at[idx, 'humidity_mean'] = zone_climate_stats[zone]['humidity_mean']
            df.at[idx, 'rainfall_mean'] = zone_climate_stats[zone]['rainfall_mean']
            df.at[idx, 'region'] = zone_climate_stats[zone]['region']

print(f"   Imputed climate data for {(df['temperature_mean'].isnull().sum() - missing_analysis['temperature_mean']):,} records")

# 2. Handle suitability and yield data (30% missing)
print("\n2. Suitability and Yield Data Handling:")

# For records with missing suitability data, we'll use crop-zone based imputation
suitability_cols = [col for col in df.columns if col.startswith('suitability_')]
yield_cols = [col for col in df.columns if col.startswith('yield_')]

# Create crop-zone statistics
crop_zone_stats = {}
for crop in df['crop_name_clean'].unique():
    crop_data = df[df['crop_name_clean'] == crop]
    for zone in crop_data['agro_ecological_zone'].dropna().unique():
        zone_crop_data = crop_data[crop_data['agro_ecological_zone'] == zone]
        if len(zone_crop_data) > 0:
            key = f"{crop}_{zone}"
            stats = {}
            for col in suitability_cols + yield_cols + ['area_hectares', 'total_suitable_area']:
                if col in df.columns:
                    stats[col] = zone_crop_data[col].mean()
            crop_zone_stats[key] = stats

print(f"   Created crop-zone statistics for {len(crop_zone_stats)} combinations")

# Impute missing suitability and yield data
imputed_count = 0
for idx, row in df.iterrows():
    if pd.isna(row['area_hectares']):
        crop = row['crop_name_clean']
        zone = row['agro_ecological_zone']
        if pd.notna(zone):
            key = f"{crop}_{zone}"
            if key in crop_zone_stats:
                stats = crop_zone_stats[key]
                for col in suitability_cols + yield_cols + ['area_hectares', 'total_suitable_area']:
                    if col in df.columns and pd.isna(row[col]):
                        df.at[idx, col] = stats[col]
                        imputed_count += 1

print(f"   Imputed suitability/yield data for {imputed_count:,} values")

# 3. Handle soil properties (16% missing)
print("\n3. Soil Properties Handling:")

# For soil properties, use crop-based imputation
soil_cols = ['pH', 'organic_matter', 'texture_class']
crop_soil_stats = {}

for crop in df['crop_name_clean'].unique():
    crop_data = df[df['crop_name_clean'] == crop]
    soil_stats = {}
    for col in soil_cols:
        if col in df.columns:
            if col == 'texture_class':
                # For categorical data, use mode
                mode_val = crop_data[col].mode()
                soil_stats[col] = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
            else:
                # For numerical data, use mean
                soil_stats[col] = crop_data[col].mean()
    crop_soil_stats[crop] = soil_stats

print(f"   Created soil statistics for {len(crop_soil_stats)} crops")

# Impute soil properties
soil_imputed = 0
for idx, row in df.iterrows():
    crop = row['crop_name_clean']
    if crop in crop_soil_stats:
        stats = crop_soil_stats[crop]
        for col in soil_cols:
            if col in df.columns and pd.isna(row[col]):
                df.at[idx, col] = stats[col]
                soil_imputed += 1

print(f"   Imputed soil properties for {soil_imputed:,} values")

# 4. Handle remaining missing values
print("\n4. Final Missing Value Handling:")

# For any remaining missing values, use global statistics
remaining_missing = df.isnull().sum()
if remaining_missing.sum() > 0:
    print("   Handling remaining missing values:")
    for col in df.columns:
        if remaining_missing[col] > 0:
            if df[col].dtype in ['object']:
                # For categorical data, use mode
                mode_val = df[col].mode()
                fill_value = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                df[col] = df[col].fillna(fill_value)
                print(f"     {col}: filled {remaining_missing[col]} values with '{fill_value}'")
            else:
                # For numerical data, use median
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                print(f"     {col}: filled {remaining_missing[col]} values with {fill_value:.2f}")

# 5. Data validation and quality checks
print(f"\n5. Data Quality Validation:")

# Check for any remaining missing values
final_missing = df.isnull().sum().sum()
print(f"   Remaining missing values: {final_missing}")

# Check for outliers in numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
outlier_summary = {}

for col in numerical_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = len(outliers)

print(f"   Outlier analysis:")
for col, count in outlier_summary.items():
    if count > 0:
        print(f"     {col}: {count} outliers detected")

# 6. Create cleaned dataset with additional features
print(f"\n6. Creating Enhanced Features:")

# Recalculate suitability scores with cleaned data
weights = {
    'suitability_very_suitable': 1.0,
    'suitability_suitable': 0.8,
    'suitability_moderately_suitable': 0.6,
    'suitability_marginally_suitable': 0.4,
    'suitability_very_marginally_suitable': 0.2
}

df['suitability_score_clean'] = 0
for col, weight in weights.items():
    if col in df.columns:
        df['suitability_score_clean'] += df[col] * weight

# Normalize by area
df['suitability_score_clean'] = df['suitability_score_clean'] / df['area_hectares']
df['suitability_score_clean'] = df['suitability_score_clean'].fillna(0)

# Create data quality flags
df['has_complete_climate'] = (~df['temperature_mean'].isnull()).astype(int)
df['has_complete_suitability'] = (~df['area_hectares'].isnull()).astype(int)
df['has_complete_soil'] = (~df['pH'].isnull()).astype(int)

# Create data completeness score
df['data_completeness'] = (df['has_complete_climate'] + 
                          df['has_complete_suitability'] + 
                          df['has_complete_soil']) / 3

# 7. Final validation
print(f"\n7. Final Dataset Validation:")

print(f"   Total records: {len(df):,}")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Data completeness score: {df['data_completeness'].mean():.3f}")
print(f"   Records with complete data: {(df['data_completeness'] == 1.0).sum():,}")

# Save cleaned dataset
output_path = '/content/drive/MyDrive/Final/data/processed/ugandan_data_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")

# Create cleaning report
cleaning_report = {
    'cleaning_timestamp': datetime.now().isoformat(),
    'original_shape': (11682, 30),
    'cleaned_shape': df.shape,
    'missing_values_before': missing_analysis.to_dict(),
    'missing_values_after': df.isnull().sum().to_dict(),
    'outlier_summary': outlier_summary,
    'data_completeness_stats': {
        'mean_completeness': float(df['data_completeness'].mean()),
        'complete_records': int((df['data_completeness'] == 1.0).sum()),
        'partial_records': int(((df['data_completeness'] > 0) & (df['data_completeness'] < 1.0)).sum())
    },
    'imputation_strategies': {
        'climate_data': 'zone_based_imputation',
        'suitability_data': 'crop_zone_based_imputation',
        'soil_data': 'crop_based_imputation',
        'remaining_data': 'global_statistics'
    }
}

# Save cleaning report
report_path = '/content/drive/MyDrive/Final/data/processed/data_cleaning_report.json'
with open(report_path, 'w') as f:
    json.dump(cleaning_report, f, indent=2)

print(f"Cleaning report saved to: {report_path}")

# Display final summary
print(f"\nData Cleaning Complete!")
print(f"=" * 50)
print(f"Final dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Data completeness: {df['data_completeness'].mean():.3f}")
print(f"Complete records: {(df['data_completeness'] == 1.0).sum():,}")

print(f"\nSample of cleaned data:")
sample_cols = ['crop_name_clean', 'zone_label', 'pH', 'organic_matter', 'suitability_score_clean', 'data_completeness']
print(df[sample_cols].head())

print(f"\nNext steps:")
print(f"  1. Convert cleaned data to knowledge graph triples")
print(f"  2. Integrate with PDF-extracted triples")
print(f"  3. Build unified knowledge graph")
print(f"  4. Train graph embeddings")
