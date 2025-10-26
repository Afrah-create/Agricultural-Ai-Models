"""
Phase 4, Cell 4: Comprehensive EDA and Knowledge Graph Visualization
This cell provides comprehensive exploratory data analysis of our knowledge graph components
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("Loading Knowledge Graph Components...")

# Load all our data components
try:
    # Load cleaned dataset
    df = pd.read_csv('/content/drive/MyDrive/Final/data/processed/ugandan_data_cleaned.csv')
    print(f"✅ Cleaned dataset loaded: {df.shape}")
    
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

def create_comprehensive_summary():
    """
    Create a comprehensive summary of our knowledge graph components
    """
    
    print("\n" + "="*60)
    print("🌾 COMPREHENSIVE KNOWLEDGE GRAPH SUMMARY")
    print("="*60)
    
    # Dataset Summary
    print(f"\n📊 DATASET COMPONENT:")
    print(f"   • Records: {len(df):,}")
    print(f"   • Unique Crops: {df['crop_name_clean'].nunique()}")
    print(f"   • Agro-ecological Zones: {df['agro_ecological_zone'].nunique()}")
    print(f"   • Generated Triples: {len(dataset_triples):,}")
    print(f"   • Data Completeness: {df['data_completeness'].mean():.3f}")
    
    # Literature Summary
    print(f"\n📚 LITERATURE COMPONENT:")
    print(f"   • PDFs Processed: {literature_analysis['total_pdfs']}")
    print(f"   • Success Rate: {literature_analysis['success_rate']:.1f}%")
    print(f"   • Soil-Crop Relationships: {literature_analysis['total_relationships']}")
    print(f"   • Management Practices: {literature_analysis['total_practices']}")
    print(f"   • Climate Requirements: {literature_analysis['total_climate']}")
    
    # Combined Knowledge Base
    total_relationships = len(dataset_triples) + literature_analysis['total_relationships']
    print(f"\n🔗 COMBINED KNOWLEDGE BASE:")
    print(f"   • Total Relationships: {total_relationships:,}")
    print(f"   • Dataset Triples: {len(dataset_triples):,} ({len(dataset_triples)/total_relationships*100:.1f}%)")
    print(f"   • Literature Triples: {literature_analysis['total_relationships']:,} ({literature_analysis['total_relationships']/total_relationships*100:.1f}%)")
    
    return {
        'dataset_records': len(df),
        'dataset_triples': len(dataset_triples),
        'literature_pdfs': literature_analysis['total_pdfs'],
        'literature_triples': literature_analysis['total_relationships'],
        'total_knowledge': total_relationships
    }

def visualize_crop_distribution():
    """
    Visualize crop distribution and characteristics
    """
    
    print(f"\n🌱 CROP DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    # Crop frequency
    crop_counts = df['crop_name_clean'].value_counts()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Crop Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Top crops by frequency
    top_crops = crop_counts.head(15)
    axes[0,0].barh(range(len(top_crops)), top_crops.values)
    axes[0,0].set_yticks(range(len(top_crops)))
    axes[0,0].set_yticklabels(top_crops.index)
    axes[0,0].set_xlabel('Number of Records')
    axes[0,0].set_title('Top 15 Crops by Record Count')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # Crop distribution pie chart
    top_10_crops = crop_counts.head(10)
    other_crops = crop_counts.iloc[10:].sum()
    pie_data = list(top_10_crops.values) + [other_crops]
    pie_labels = list(top_10_crops.index) + ['Others']
    
    axes[0,1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('Crop Distribution (Top 10 + Others)')
    
    # Average suitability by crop
    crop_suitability = df.groupby('crop_name_clean')['overall_suitability'].mean().sort_values(ascending=False)
    top_suitable = crop_suitability.head(10)
    
    axes[1,0].bar(range(len(top_suitable)), top_suitable.values)
    axes[1,0].set_xticks(range(len(top_suitable)))
    axes[1,0].set_xticklabels(top_suitable.index, rotation=45, ha='right')
    axes[1,0].set_ylabel('Average Overall Suitability')
    axes[1,0].set_title('Top 10 Crops by Average Suitability')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Yield potential by crop
    crop_yield = df.groupby('crop_name_clean')['yield_score'].mean().sort_values(ascending=False)
    top_yield = crop_yield.head(10)
    
    axes[1,1].bar(range(len(top_yield)), top_yield.values)
    axes[1,1].set_xticks(range(len(top_yield)))
    axes[1,1].set_xticklabels(top_yield.index, rotation=45, ha='right')
    axes[1,1].set_ylabel('Average Yield Score')
    axes[1,1].set_title('Top 10 Crops by Yield Potential')
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print crop statistics
    print(f"Total unique crops: {len(crop_counts)}")
    print(f"Most common crop: {crop_counts.index[0]} ({crop_counts.iloc[0]:,} records)")
    print(f"Average records per crop: {crop_counts.mean():.1f}")
    print(f"Crop with highest suitability: {crop_suitability.index[0]} ({crop_suitability.iloc[0]:.3f})")
    print(f"Crop with highest yield potential: {crop_yield.index[0]} ({crop_yield.iloc[0]:.3f})")

def visualize_soil_properties():
    """
    Visualize soil properties and their relationships
    """
    
    print(f"\n🌍 SOIL PROPERTIES ANALYSIS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Soil Properties Analysis', fontsize=16, fontweight='bold')
    
    # pH distribution
    axes[0,0].hist(df['pH'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('pH Level')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('pH Distribution')
    axes[0,0].grid(alpha=0.3)
    
    # Organic matter distribution
    axes[0,1].hist(df['organic_matter'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_xlabel('Organic Matter (%)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Organic Matter Distribution')
    axes[0,1].grid(alpha=0.3)
    
    # Nutrient levels
    nutrients = ['nitrogen', 'phosphorus', 'potassium']
    nutrient_data = [df[nutrient] for nutrient in nutrients]
    
    axes[0,2].boxplot(nutrient_data, labels=nutrients)
    axes[0,2].set_ylabel('Nutrient Level (kg/ha)')
    axes[0,2].set_title('Nutrient Levels Distribution')
    axes[0,2].grid(alpha=0.3)
    
    # pH vs Suitability
    axes[1,0].scatter(df['pH'], df['overall_suitability'], alpha=0.5, s=1)
    axes[1,0].set_xlabel('pH Level')
    axes[1,0].set_ylabel('Overall Suitability')
    axes[1,0].set_title('pH vs Overall Suitability')
    axes[1,0].grid(alpha=0.3)
    
    # Organic matter vs Suitability
    axes[1,1].scatter(df['organic_matter'], df['overall_suitability'], alpha=0.5, s=1)
    axes[1,1].set_xlabel('Organic Matter (%)')
    axes[1,1].set_ylabel('Overall Suitability')
    axes[1,1].set_title('Organic Matter vs Overall Suitability')
    axes[1,1].grid(alpha=0.3)
    
    # Texture class distribution
    texture_counts = df['texture_class'].value_counts()
    axes[1,2].bar(range(len(texture_counts)), texture_counts.values)
    axes[1,2].set_xticks(range(len(texture_counts)))
    axes[1,2].set_xticklabels(texture_counts.index, rotation=45, ha='right')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Soil Texture Distribution')
    axes[1,2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print soil statistics
    print(f"pH range: {df['pH'].min():.2f} - {df['pH'].max():.2f} (mean: {df['pH'].mean():.2f})")
    print(f"Organic matter range: {df['organic_matter'].min():.2f}% - {df['organic_matter'].max():.2f}% (mean: {df['organic_matter'].mean():.2f}%)")
    print(f"Nitrogen range: {df['nitrogen'].min():.1f} - {df['nitrogen'].max():.1f} kg/ha (mean: {df['nitrogen'].mean():.1f})")
    print(f"Phosphorus range: {df['phosphorus'].min():.1f} - {df['phosphorus'].max():.1f} kg/ha (mean: {df['phosphorus'].mean():.1f})")
    print(f"Potassium range: {df['potassium'].min():.1f} - {df['potassium'].max():.1f} kg/ha (mean: {df['potassium'].mean():.1f})")

def visualize_agro_ecological_zones():
    """
    Visualize agro-ecological zones and their characteristics
    """
    
    print(f"\n🗺️ AGRO-ECOLOGICAL ZONES ANALYSIS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Agro-Ecological Zones Analysis', fontsize=16, fontweight='bold')
    
    # Zone distribution
    zone_counts = df['agro_ecological_zone'].value_counts().sort_index()
    axes[0,0].bar(zone_counts.index, zone_counts.values)
    axes[0,0].set_xlabel('Agro-Ecological Zone')
    axes[0,0].set_ylabel('Number of Records')
    axes[0,0].set_title('Records per Agro-Ecological Zone')
    axes[0,0].grid(alpha=0.3)
    
    # Average suitability by zone
    zone_suitability = df.groupby('agro_ecological_zone')['overall_suitability'].mean().sort_index()
    axes[0,1].bar(zone_suitability.index, zone_suitability.values)
    axes[0,1].set_xlabel('Agro-Ecological Zone')
    axes[0,1].set_ylabel('Average Overall Suitability')
    axes[0,1].set_title('Average Suitability by Zone')
    axes[0,1].grid(alpha=0.3)
    
    # Climate variables by zone
    climate_vars = ['temperature_mean', 'humidity_mean', 'rainfall_mean']
    zone_climate = df.groupby('agro_ecological_zone')[climate_vars].mean()
    
    x = np.arange(len(zone_climate.index))
    width = 0.25
    
    for i, var in enumerate(climate_vars):
        axes[1,0].bar(x + i*width, zone_climate[var], width, label=var.replace('_', ' ').title())
    
    axes[1,0].set_xlabel('Agro-Ecological Zone')
    axes[1,0].set_ylabel('Average Value')
    axes[1,0].set_title('Climate Variables by Zone')
    axes[1,0].set_xticks(x + width)
    axes[1,0].set_xticklabels(zone_climate.index)
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    # Crop diversity by zone
    zone_crop_diversity = df.groupby('agro_ecological_zone')['crop_name_clean'].nunique().sort_index()
    axes[1,1].bar(zone_crop_diversity.index, zone_crop_diversity.values)
    axes[1,1].set_xlabel('Agro-Ecological Zone')
    axes[1,1].set_ylabel('Number of Unique Crops')
    axes[1,1].set_title('Crop Diversity by Zone')
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print zone statistics
    print(f"Total agro-ecological zones: {len(zone_counts)}")
    print(f"Zone with most records: {zone_counts.index[0]} ({zone_counts.iloc[0]:,} records)")
    print(f"Zone with highest suitability: {zone_suitability.idxmax()} ({zone_suitability.max():.3f})")
    print(f"Zone with highest crop diversity: {zone_crop_diversity.idxmax()} ({zone_crop_diversity.max()} crops)")

def visualize_knowledge_graph_structure():
    """
    Visualize the structure of our knowledge graph
    """
    
    print(f"\n🔗 KNOWLEDGE GRAPH STRUCTURE ANALYSIS")
    print("-" * 40)
    
    # Analyze dataset triples
    triple_types = Counter([triple['triple_type'] for triple in dataset_triples])
    predicates = Counter([triple['predicate'] for triple in dataset_triples])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Knowledge Graph Structure Analysis', fontsize=16, fontweight='bold')
    
    # Triple types distribution
    axes[0,0].barh(range(len(triple_types)), list(triple_types.values()))
    axes[0,0].set_yticks(range(len(triple_types)))
    axes[0,0].set_yticklabels(list(triple_types.keys()))
    axes[0,0].set_xlabel('Number of Triples')
    axes[0,0].set_title('Dataset Triple Types Distribution')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # Top predicates
    top_predicates = dict(predicates.most_common(10))
    axes[0,1].barh(range(len(top_predicates)), list(top_predicates.values()))
    axes[0,1].set_yticks(range(len(top_predicates)))
    axes[0,1].set_yticklabels(list(top_predicates.keys()))
    axes[0,1].set_xlabel('Number of Triples')
    axes[0,1].set_title('Top 10 Predicates')
    axes[0,1].grid(axis='x', alpha=0.3)
    
    # Literature vs Dataset comparison
    categories = ['Dataset Triples', 'Literature Relationships', 'Literature Practices', 'Literature Climate']
    values = [
        len(dataset_triples),
        literature_analysis['total_relationships'],
        literature_analysis['total_practices'],
        literature_analysis['total_climate']
    ]
    
    axes[1,0].bar(categories, values)
    axes[1,0].set_ylabel('Number of Items')
    axes[1,0].set_title('Knowledge Base Components Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(alpha=0.3)
    
    # Data completeness
    completeness_data = [
        df['has_complete_climate'].sum(),
        df['has_complete_suitability'].sum(),
        df['has_complete_soil'].sum(),
        len(df)
    ]
    completeness_labels = ['Complete Climate', 'Complete Suitability', 'Complete Soil', 'Total Records']
    
    axes[1,1].bar(completeness_labels, completeness_data)
    axes[1,1].set_ylabel('Number of Records')
    axes[1,1].set_title('Data Completeness by Category')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print knowledge graph statistics
    print(f"Total dataset triples: {len(dataset_triples):,}")
    print(f"Total literature relationships: {literature_analysis['total_relationships']:,}")
    print(f"Total literature practices: {literature_analysis['total_practices']:,}")
    print(f"Total literature climate requirements: {literature_analysis['total_climate']:,}")
    print(f"Data completeness: {df['data_completeness'].mean():.3f}")

def create_network_visualization():
    """
    Create a network visualization of crop-soil relationships
    """
    
    print(f"\n🕸️ NETWORK VISUALIZATION")
    print("-" * 40)
    
    # Create a sample network for visualization (top crops and their relationships)
    G = nx.Graph()
    
    # Add top 10 crops as nodes
    top_crops = df['crop_name_clean'].value_counts().head(10).index.tolist()
    
    for crop in top_crops:
        G.add_node(crop, node_type='crop', size=df[df['crop_name_clean']==crop].shape[0])
    
    # Add soil properties as nodes
    soil_props = ['pH', 'organic_matter', 'nitrogen', 'phosphorus', 'potassium']
    for prop in soil_props:
        G.add_node(prop, node_type='soil_property', size=1000)
    
    # Add agro-ecological zones as nodes
    top_zones = df['agro_ecological_zone'].value_counts().head(5).index.tolist()
    for zone in top_zones:
        G.add_node(f'Zone_{zone}', node_type='zone', size=500)
    
    # Add edges based on relationships
    for _, row in df.iterrows():
        crop = row['crop_name_clean']
        if crop in top_crops:
            # Crop to soil property relationships
            for prop in soil_props:
                if pd.notna(row[prop]):
                    G.add_edge(crop, prop, weight=1)
            
            # Crop to zone relationships
            if pd.notna(row['agro_ecological_zone']):
                zone_node = f'Zone_{row["agro_ecological_zone"]}'
                if zone_node in G.nodes():
                    G.add_edge(crop, zone_node, weight=1)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Position nodes
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes by type
    crop_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'crop']
    soil_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'soil_property']
    zone_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'zone']
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=crop_nodes, node_color='lightblue', 
                          node_size=800, alpha=0.8, label='Crops')
    nx.draw_networkx_nodes(G, pos, nodelist=soil_nodes, node_color='lightgreen', 
                          node_size=600, alpha=0.8, label='Soil Properties')
    nx.draw_networkx_nodes(G, pos, nodelist=zone_nodes, node_color='lightcoral', 
                          node_size=400, alpha=0.8, label='Agro-Ecological Zones')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title('Crop-Soil-Zone Relationship Network\n(Top 10 Crops, Soil Properties, Top 5 Zones)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print network statistics
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.3f}")
    print(f"Average clustering: {nx.average_clustering(G):.3f}")

# Execute comprehensive EDA
print("Starting Comprehensive EDA...")

# Create summary
summary = create_comprehensive_summary()

# Visualize components
visualize_crop_distribution()
visualize_soil_properties()
visualize_agro_ecological_zones()
visualize_knowledge_graph_structure()
create_network_visualization()

print(f"\n🎉 COMPREHENSIVE EDA COMPLETE!")
print(f"="*60)
print(f"📊 Dataset: {summary['dataset_records']:,} records → {summary['dataset_triples']:,} triples")
print(f"📚 Literature: {summary['literature_pdfs']} PDFs → {summary['literature_triples']:,} relationships")
print(f"🔗 Total Knowledge Base: {summary['total_knowledge']:,} relationships")
print(f"🌾 Crops: {df['crop_name_clean'].nunique()} unique")
print(f"🗺️ Zones: {df['agro_ecological_zone'].nunique()} agro-ecological zones")
print(f"📈 Data Quality: {df['data_completeness'].mean():.3f} completeness score")

print(f"\nNext steps:")
print(f"  1. Integrate PDF and dataset triples")
print(f"  2. Build unified knowledge graph")
print(f"  3. Train graph embeddings")
print(f"  4. Develop RAG pipeline")
