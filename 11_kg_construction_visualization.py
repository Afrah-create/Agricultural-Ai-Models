"""
Phase 4, Cell 6: Knowledge Graph Construction and Visualization
This cell builds the NetworkX knowledge graph and creates comprehensive visualizations
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

print("Loading Unified Knowledge Graph...")

# Load unified knowledge graph
try:
    with open('/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph.json', 'r') as f:
        unified_triples = json.load(f)
    print(f"✅ Unified knowledge graph loaded: {len(unified_triples):,} triples")
    
    with open('/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph_analysis.json', 'r') as f:
        analysis = json.load(f)
    print(f"✅ Analysis loaded")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("Please run the integration cell first")

def build_knowledge_graph(unified_triples):
    """
    Build NetworkX knowledge graph from unified triples
    """
    
    print("Building NetworkX knowledge graph...")
    
    G = nx.MultiDiGraph()
    
    # Add nodes and edges
    for triple in unified_triples:
        subject = triple.get('subject', '')
        predicate = triple.get('predicate', '')
        object_node = triple.get('object', '')
        
        if subject and object_node:
            # Add nodes with attributes
            G.add_node(subject, **{k: v for k, v in triple.items() if k not in ['subject', 'predicate', 'object']})
            G.add_node(object_node, **{k: v for k, v in triple.items() if k not in ['subject', 'predicate', 'object']})
            
            # Add edge with attributes
            G.add_edge(subject, object_node, 
                      predicate=predicate,
                      **{k: v for k, v in triple.items() if k not in ['subject', 'predicate', 'object']})
    
    print(f"Knowledge graph built:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Density: {nx.density(G):.4f}")
    
    return G

def analyze_graph_structure(G):
    """
    Analyze the structure of the knowledge graph
    """
    
    print(f"\nGraph Structure Analysis:")
    print(f"=" * 40)
    
    # Basic metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    print(f"Nodes: {num_nodes:,}")
    print(f"Edges: {num_edges:,}")
    print(f"Density: {density:.4f}")
    
    # Node types
    node_types = defaultdict(int)
    for node in G.nodes():
        if 'crop/' in node:
            node_types['crop'] += 1
        elif 'soil_property/' in node:
            node_types['soil_property'] += 1
        elif 'agro_ecological_zone/' in node:
            node_types['agro_ecological_zone'] += 1
        elif 'management_practice/' in node:
            node_types['management_practice'] += 1
        elif 'climate_zone/' in node:
            node_types['climate_zone'] += 1
        else:
            node_types['other'] += 1
    
    print(f"\nNode Types:")
    for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node_type}: {count:,}")
    
    # Edge types (predicates)
    edge_types = Counter()
    for u, v, data in G.edges(data=True):
        predicate = data.get('predicate', 'unknown')
        edge_types[predicate] += 1
    
    print(f"\nTop Edge Types (Predicates):")
    for predicate, count in edge_types.most_common(10):
        print(f"  {predicate}: {count:,}")
    
    # Connectivity metrics
    if nx.is_weakly_connected(G):
        print(f"\nConnectivity:")
        print(f"  Weakly connected: Yes")
        print(f"  Strongly connected: {nx.is_strongly_connected(G)}")
        print(f"  Number of weakly connected components: {nx.number_weakly_connected_components(G)}")
        print(f"  Number of strongly connected components: {nx.number_strongly_connected_components(G)}")
    
    # Centrality metrics for top nodes
    print(f"\nCentrality Analysis:")
    
    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top 10 nodes by degree centrality:")
    for node, centrality in top_degree:
        print(f"    {node}: {centrality:.4f}")
    
    # Betweenness centrality (sample for large graphs)
    if num_nodes < 1000:
        betweenness_centrality = nx.betweenness_centrality(G)
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 nodes by betweenness centrality:")
        for node, centrality in top_betweenness:
            print(f"    {node}: {centrality:.4f}")
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'node_types': dict(node_types),
        'edge_types': dict(edge_types),
        'degree_centrality': degree_centrality
    }

def visualize_graph_overview(G, analysis):
    """
    Create overview visualizations of the knowledge graph
    """
    
    print(f"\nCreating graph overview visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Knowledge Graph Overview', fontsize=16, fontweight='bold')
    
    # Node types distribution
    node_types = analysis['node_types']
    axes[0,0].bar(range(len(node_types)), list(node_types.values()))
    axes[0,0].set_xticks(range(len(node_types)))
    axes[0,0].set_xticklabels(list(node_types.keys()), rotation=45, ha='right')
    axes[0,0].set_ylabel('Number of Nodes')
    axes[0,0].set_title('Node Types Distribution')
    axes[0,0].grid(alpha=0.3)
    
    # Top predicates
    edge_types = analysis['edge_types']
    top_predicates = dict(list(edge_types.items())[:10])
    axes[0,1].barh(range(len(top_predicates)), list(top_predicates.values()))
    axes[0,1].set_yticks(range(len(top_predicates)))
    axes[0,1].set_yticklabels(list(top_predicates.keys()))
    axes[0,1].set_xlabel('Number of Edges')
    axes[0,1].set_title('Top 10 Predicates')
    axes[0,1].grid(alpha=0.3)
    
    # Degree distribution
    degrees = [G.degree(node) for node in G.nodes()]
    axes[1,0].hist(degrees, bins=30, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Degree')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Degree Distribution')
    axes[1,0].grid(alpha=0.3)
    
    # Top nodes by degree centrality
    degree_centrality = analysis['degree_centrality']
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_nodes = [node.split('/')[-1] for node, _ in top_degree]
    top_values = [centrality for _, centrality in top_degree]
    
    axes[1,1].barh(range(len(top_nodes)), top_values)
    axes[1,1].set_yticks(range(len(top_nodes)))
    axes[1,1].set_yticklabels(top_nodes)
    axes[1,1].set_xlabel('Degree Centrality')
    axes[1,1].set_title('Top 10 Nodes by Degree Centrality')
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_crop_centered_subgraph(G, top_crops=5):
    """
    Create a subgraph centered around top crops
    """
    
    print(f"\nCreating crop-centered subgraph...")
    
    # Get top crops by degree
    degree_centrality = nx.degree_centrality(G)
    crop_nodes = [node for node in G.nodes() if 'crop/' in node]
    crop_degrees = [(node, degree_centrality[node]) for node in crop_nodes]
    top_crop_nodes = [node for node, _ in sorted(crop_degrees, key=lambda x: x[1], reverse=True)[:top_crops]]
    
    # Create subgraph with top crops and their neighbors
    subgraph_nodes = set(top_crop_nodes)
    for crop in top_crop_nodes:
        # Add neighbors within 2 hops
        neighbors = list(G.neighbors(crop))
        subgraph_nodes.update(neighbors)
        for neighbor in neighbors:
            subgraph_nodes.update(list(G.neighbors(neighbor)))
    
    subgraph = G.subgraph(subgraph_nodes)
    
    print(f"Subgraph created:")
    print(f"  Nodes: {subgraph.number_of_nodes()}")
    print(f"  Edges: {subgraph.number_of_edges()}")
    
    return subgraph, top_crop_nodes

def visualize_crop_network(subgraph, top_crop_nodes):
    """
    Visualize the crop-centered network
    """
    
    print(f"\nVisualizing crop-centered network...")
    
    plt.figure(figsize=(16, 12))
    
    # Create layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50)
    
    # Separate nodes by type
    crop_nodes = [n for n in subgraph.nodes() if 'crop/' in n]
    soil_nodes = [n for n in subgraph.nodes() if 'soil_property/' in n]
    zone_nodes = [n for n in subgraph.nodes() if 'agro_ecological_zone/' in n]
    management_nodes = [n for n in subgraph.nodes() if 'management_practice/' in n]
    other_nodes = [n for n in subgraph.nodes() if n not in crop_nodes + soil_nodes + zone_nodes + management_nodes]
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Draw nodes by type
    nx.draw_networkx_nodes(subgraph, pos, nodelist=crop_nodes, 
                          node_color='lightblue', node_size=800, alpha=0.8, label='Crops')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=soil_nodes, 
                          node_color='lightgreen', node_size=600, alpha=0.8, label='Soil Properties')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=zone_nodes, 
                          node_color='lightcoral', node_size=500, alpha=0.8, label='Agro-Ecological Zones')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=management_nodes, 
                          node_color='lightyellow', node_size=400, alpha=0.8, label='Management Practices')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=other_nodes, 
                          node_color='lightgray', node_size=300, alpha=0.8, label='Other')
    
    # Highlight top crops
    nx.draw_networkx_nodes(subgraph, pos, nodelist=top_crop_nodes, 
                          node_color='darkblue', node_size=1000, alpha=0.9)
    
    # Draw labels for important nodes
    important_nodes = top_crop_nodes + soil_nodes[:5] + zone_nodes[:3]
    labels = {node: node.split('/')[-1] for node in important_nodes}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')
    
    plt.title('Crop-Centered Knowledge Graph\n(Top 5 Crops and Their Relationships)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_soil_crop_relationship_matrix(G):
    """
    Create a matrix showing soil-crop relationships
    """
    
    print(f"\nCreating soil-crop relationship matrix...")
    
    # Get all crops and soil properties
    crops = [node for node in G.nodes() if 'crop/' in node]
    soil_props = [node for node in G.nodes() if 'soil_property/' in node]
    
    # Create relationship matrix
    matrix = np.zeros((len(crops), len(soil_props)))
    
    for i, crop in enumerate(crops):
        for j, soil_prop in enumerate(soil_props):
            # Count edges between crop and soil property
            if G.has_edge(crop, soil_prop):
                matrix[i, j] += 1
            if G.has_edge(soil_prop, crop):
                matrix[i, j] += 1
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Crop names for y-axis
    crop_names = [crop.split('/')[-1] for crop in crops]
    soil_names = [soil.split('/')[-1] for soil in soil_props]
    
    # Create heatmap
    sns.heatmap(matrix, 
                xticklabels=soil_names, 
                yticklabels=crop_names,
                cmap='YlOrRd', 
                annot=True, 
                fmt='.0f',
                cbar_kws={'label': 'Number of Relationships'})
    
    plt.title('Soil-Crop Relationship Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Soil Properties')
    plt.ylabel('Crops')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return matrix, crop_names, soil_names

def analyze_crop_profiles(G):
    """
    Analyze detailed profiles for top crops
    """
    
    print(f"\nAnalyzing crop profiles...")
    
    # Get top crops by degree centrality
    degree_centrality = nx.degree_centrality(G)
    crop_nodes = [node for node in G.nodes() if 'crop/' in node]
    crop_degrees = [(node, degree_centrality[node]) for node in crop_nodes]
    top_crops = sorted(crop_degrees, key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\nTop 5 Crops Analysis:")
    print("-" * 30)
    
    for i, (crop_node, centrality) in enumerate(top_crops):
        crop_name = crop_node.split('/')[-1]
        print(f"\n{i+1}. {crop_name} (Centrality: {centrality:.4f})")
        
        # Get neighbors
        neighbors = list(G.neighbors(crop_node))
        predecessors = list(G.predecessors(crop_node))
        
        print(f"   Total connections: {len(neighbors) + len(predecessors)}")
        
        # Categorize connections
        soil_connections = [n for n in neighbors + predecessors if 'soil_property/' in n]
        zone_connections = [n for n in neighbors + predecessors if 'agro_ecological_zone/' in n]
        management_connections = [n for n in neighbors + predecessors if 'management_practice/' in n]
        
        print(f"   Soil property connections: {len(soil_connections)}")
        print(f"   Zone connections: {len(zone_connections)}")
        print(f"   Management connections: {len(management_connections)}")
        
        # Show specific relationships
        if soil_connections:
            print(f"   Connected soil properties: {[n.split('/')[-1] for n in soil_connections[:3]]}")
        if zone_connections:
            print(f"   Connected zones: {[n.split('/')[-1] for n in zone_connections[:3]]}")

def save_graph_metrics(G, analysis):
    """
    Save graph metrics and structure
    """
    
    print(f"\nSaving graph metrics...")
    
    # Basic metrics
    metrics = {
        'basic_metrics': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G),
            'num_components': nx.number_weakly_connected_components(G)
        },
        'node_types': analysis['node_types'],
        'edge_types': analysis['edge_types'],
        'top_nodes_by_degree': list(analysis['degree_centrality'].items())[:20],
        'created_at': datetime.now().isoformat()
    }
    
    # Save metrics
    metrics_path = '/content/drive/MyDrive/Final/data/processed/graph_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Graph metrics saved to: {metrics_path}")
    
    # Save graph structure (simplified for large graphs)
    if G.number_of_nodes() < 1000:
        graph_data = {
            'nodes': list(G.nodes(data=True)),
            'edges': list(G.edges(data=True))
        }
        
        graph_path = '/content/drive/MyDrive/Final/data/processed/graph_structure.json'
        with open(graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph structure saved to: {graph_path}")
    else:
        print("Graph too large to save structure (nodes > 1000)")

# Execute knowledge graph construction and visualization
print("Starting Knowledge Graph Construction and Visualization...")

# Build knowledge graph
G = build_knowledge_graph(unified_triples)

# Analyze graph structure
graph_analysis = analyze_graph_structure(G)

# Create visualizations
visualize_graph_overview(G, graph_analysis)

# Create crop-centered subgraph
subgraph, top_crop_nodes = create_crop_centered_subgraph(G)

# Visualize crop network
visualize_crop_network(subgraph, top_crop_nodes)

# Create soil-crop relationship matrix
matrix, crop_names, soil_names = create_soil_crop_relationship_matrix(G)

# Analyze crop profiles
analyze_crop_profiles(G)

# Save metrics
save_graph_metrics(G, graph_analysis)

print(f"\n🎉 Knowledge Graph Construction and Visualization Complete!")
print(f"=" * 60)
print(f"📊 Graph Statistics:")
print(f"   Nodes: {G.number_of_nodes():,}")
print(f"   Edges: {G.number_of_edges():,}")
print(f"   Density: {nx.density(G):.4f}")
print(f"   Connected: {nx.is_weakly_connected(G)}")
print(f"   Components: {nx.number_weakly_connected_components(G)}")

print(f"\n🌾 Top Crops by Connectivity:")
degree_centrality = nx.degree_centrality(G)
crop_nodes = [node for node in G.nodes() if 'crop/' in node]
crop_degrees = [(node.split('/')[-1], degree_centrality[node]) for node in crop_nodes]
top_crops = sorted(crop_degrees, key=lambda x: x[1], reverse=True)[:5]

for i, (crop, centrality) in enumerate(top_crops):
    print(f"   {i+1}. {crop}: {centrality:.4f}")

print(f"\nNext steps:")
print(f"  1. Train graph embeddings")
print(f"  2. Develop RAG pipeline")
print(f"  3. Create crop recommendation system")
print(f"  4. Build LLM fine-tuning dataset")
