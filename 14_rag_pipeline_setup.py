"""
Phase 6, Cell 1: RAG Pipeline Setup and Configuration
This cell sets up the Retrieval-Augmented Generation pipeline for evidence-backed crop recommendations
"""

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Import Gemini API for LLM integration
import google.generativeai as genai

# Model class definitions (matching the training script)
class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super(TransEModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        h = F.normalize(self.entity_embeddings(head), p=2, dim=1)
        r = F.normalize(self.relation_embeddings(relation), p=2, dim=1)
        t = F.normalize(self.entity_embeddings(tail), p=2, dim=1)
        
        score = torch.norm(h + r - t, p=2, dim=1)
        return score

class DistMultModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super(DistMultModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        score = torch.sum(h * r * t, dim=1)
        return score

class ComplExModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super(ComplExModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings_real.weight)
        nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_real.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)
    
    def forward(self, head, relation, tail):
        h_real = self.entity_embeddings_real(head)
        h_imag = self.entity_embeddings_imag(head)
        r_real = self.relation_embeddings_real(relation)
        r_imag = self.relation_embeddings_imag(relation)
        t_real = self.entity_embeddings_real(tail)
        t_imag = self.entity_embeddings_imag(tail)
        
        score = torch.sum(
            h_real * r_real * t_real +
            h_imag * r_real * t_imag +
            h_real * r_imag * t_imag -
            h_imag * r_imag * t_real,
            dim=1
        )
        return score

class GCNModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, hidden_dim=200):
        super(GCNModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self.gcn1 = nn.Linear(embedding_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        h_gcn = F.relu(self.gcn1(h))
        h_gcn = self.gcn2(h_gcn)
        
        t_gcn = F.relu(self.gcn1(t))
        t_gcn = self.gcn2(t_gcn)
        
        score = torch.sum(h_gcn * r * t_gcn, dim=1)
        return score

class GraphSAGEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, hidden_dim=200):
        super(GraphSAGEModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self.sage1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.sage2 = nn.Linear(hidden_dim, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        h_concat = torch.cat([h, r], dim=1)
        h_sage = F.relu(self.sage1(h_concat))
        h_sage = self.sage2(h_sage)
        
        t_concat = torch.cat([t, r], dim=1)
        t_sage = F.relu(self.sage1(t_concat))
        t_sage = self.sage2(t_sage)
        
        score = torch.sum(h_sage * t_sage, dim=1)
        return score

# Configure Gemini API
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("✅ Gemini API configured successfully!")
except Exception as e:
    print(f"❌ Gemini API configuration failed: {e}")
    print("Trying alternative model...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Using Gemini 1.5 Flash model")
    except Exception as e2:
        print(f"❌ Alternative model also failed: {e2}")
        print("Please ensure GEMINI_API_KEY is set in Colab secrets")

print("Loading trained models and knowledge graph...")

# Load the best model metadata
try:
    with open('/content/drive/MyDrive/Final/data/processed/trained_models/best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    print(f"✅ Best model loaded: {best_model_info['model_name']}")
    print(f"   F1 Score: {best_model_info['f1_score']:.4f}")
except Exception as e:
    print(f"❌ Error loading best model info: {e}")
    # Fallback to DistMult if GCN not available
    best_model_info = {
        'model_name': 'DistMult',
        'f1_score': 0.8607,
        'model_path': '/content/drive/MyDrive/Final/data/processed/trained_models/distmult_model.pth'
    }

# Load model metadata
try:
    with open('/content/drive/MyDrive/Final/data/processed/trained_models/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    print(f"✅ Model metadata loaded: {model_metadata['num_entities']} entities, {model_metadata['num_relations']} relations")
except Exception as e:
    print(f"❌ Error loading model metadata: {e}")

# Load unified knowledge graph
try:
    with open('/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph.json', 'r') as f:
        unified_triples = json.load(f)
    print(f"✅ Unified knowledge graph loaded: {len(unified_triples)} triples")
except Exception as e:
    print(f"❌ Error loading knowledge graph: {e}")

# Load processed dataset for additional context
try:
    df = pd.read_csv('/content/drive/MyDrive/Final/data/processed/ugandan_data_cleaned.csv')
    print(f"✅ Processed dataset loaded: {len(df)} records")
except Exception as e:
    print(f"❌ Error loading processed dataset: {e}")

class GraphEmbeddingRetriever:
    """
    Graph embedding-based retrieval system for RAG pipeline
    """
    
    def __init__(self, model_path, metadata_path, triples_data):
        self.triples_data = triples_data
        self.entity_to_id = model_metadata['entity_to_id']
        self.relation_to_id = model_metadata['relation_to_id']
        self.id_to_entity = model_metadata['id_to_entity']
        self.id_to_relation = model_metadata['id_to_relation']
        self.num_entities = model_metadata['num_entities']
        self.num_relations = model_metadata['num_relations']
        self.embedding_dim = model_metadata['embedding_dim']
        
        # Load the best model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Create entity embeddings cache
        self.entity_embeddings = self._create_entity_embeddings()
        
        # Create triple representations
        self.triple_embeddings = self._create_triple_embeddings()
        
        print(f"✅ GraphEmbeddingRetriever initialized with {len(self.triple_embeddings)} triple embeddings")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        if best_model_info['model_name'] == 'GCN':
            model = GCNModel(self.num_entities, self.num_relations, self.embedding_dim)
        elif best_model_info['model_name'] == 'DistMult':
            model = DistMultModel(self.num_entities, self.num_relations, self.embedding_dim)
        elif best_model_info['model_name'] == 'TransE':
            model = TransEModel(self.num_entities, self.num_relations, self.embedding_dim)
        elif best_model_info['model_name'] == 'ComplEx':
            model = ComplExModel(self.num_entities, self.num_relations, self.embedding_dim)
        elif best_model_info['model_name'] == 'GraphSAGE':
            model = GraphSAGEModel(self.num_entities, self.num_relations, self.embedding_dim)
        else:
            # Default to DistMult
            model = DistMultModel(self.num_entities, self.num_relations, self.embedding_dim)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model
    
    def _create_entity_embeddings(self):
        """Create entity embeddings for similarity search"""
        with torch.no_grad():
            entity_ids = torch.arange(self.num_entities)
            embeddings = self.model.entity_embeddings(entity_ids)
            return embeddings.numpy()
    
    def _create_triple_embeddings(self):
        """Create embeddings for all triples in the knowledge graph"""
        triple_embeddings = []
        
        for triple in self.triples_data:
            head_id = self.entity_to_id.get(triple['subject'], 0)
            relation_id = self.relation_to_id.get(triple['predicate'], 0)
            tail_id = self.entity_to_id.get(triple['object'], 0)
            
            # Get embeddings
            head_emb = self.entity_embeddings[head_id]
            tail_emb = self.entity_embeddings[tail_id]
            
            # Create triple representation (concatenation of head and tail)
            triple_emb = np.concatenate([head_emb, tail_emb])
            triple_embeddings.append({
                'triple': triple,
                'embedding': triple_emb,
                'head_id': head_id,
                'relation_id': relation_id,
                'tail_id': tail_id
            })
        
        return triple_embeddings
    
    def retrieve_relevant_triples(self, query_text, top_k=10):
        """
        Retrieve most relevant triples based on query similarity
        """
        # Simple keyword-based retrieval for now
        # In a more sophisticated system, we'd use query embedding
        
        query_lower = query_text.lower()
        relevant_triples = []
        
        for triple_emb in self.triple_embeddings:
            triple = triple_emb['triple']
            
            # Check if query terms appear in triple components
            subject_match = any(term in triple['subject'].lower() for term in query_lower.split())
            object_match = any(term in triple['object'].lower() for term in query_lower.split())
            predicate_match = any(term in triple['predicate'].lower() for term in query_lower.split())
            
            if subject_match or object_match or predicate_match:
                relevant_triples.append(triple_emb)
        
        # Sort by relevance (simple scoring)
        relevant_triples.sort(key=lambda x: self._calculate_relevance_score(x['triple'], query_text), reverse=True)
        
        return relevant_triples[:top_k]
    
    def _calculate_relevance_score(self, triple, query):
        """Calculate relevance score for a triple given a query"""
        query_terms = set(query.lower().split())
        
        subject_terms = set(triple['subject'].lower().split())
        object_terms = set(triple['object'].lower().split())
        predicate_terms = set(triple['predicate'].lower().split())
        
        # Calculate overlap
        subject_overlap = len(query_terms.intersection(subject_terms))
        object_overlap = len(query_terms.intersection(object_terms))
        predicate_overlap = len(query_terms.intersection(predicate_terms))
        
        # Weighted score
        score = subject_overlap * 2 + object_overlap * 2 + predicate_overlap * 1
        
        return score

class AgriculturalRAGPipeline:
    """
    Main RAG pipeline for agricultural recommendations
    """
    
    def __init__(self, retriever, llm_model):
        self.retriever = retriever
        self.llm_model = llm_model
        
        # Agricultural domain knowledge
        self.crop_constraints = {
            'maize': {'min_temp': 15, 'max_temp': 30, 'min_rainfall': 500, 'max_rainfall': 1200, 'optimal_ph': (6.0, 7.0)},
            'rice': {'min_temp': 20, 'max_temp': 35, 'min_rainfall': 1000, 'max_rainfall': 2000, 'optimal_ph': (5.5, 6.5)},
            'beans': {'min_temp': 18, 'max_temp': 28, 'min_rainfall': 400, 'max_rainfall': 1000, 'optimal_ph': (6.0, 7.5)},
            'cassava': {'min_temp': 20, 'max_temp': 35, 'min_rainfall': 600, 'max_rainfall': 1500, 'optimal_ph': (5.5, 7.0)},
            'sweet_potato': {'min_temp': 18, 'max_temp': 30, 'min_rainfall': 500, 'max_rainfall': 1200, 'optimal_ph': (5.5, 6.5)},
            'banana': {'min_temp': 20, 'max_temp': 35, 'min_rainfall': 1000, 'max_rainfall': 2000, 'optimal_ph': (6.0, 7.0)},
            'coffee': {'min_temp': 15, 'max_temp': 25, 'min_rainfall': 1200, 'max_rainfall': 2000, 'optimal_ph': (6.0, 6.5)},
            'cotton': {'min_temp': 20, 'max_temp': 35, 'min_rainfall': 500, 'max_rainfall': 1200, 'optimal_ph': (6.0, 7.0)},
            'sugarcane': {'min_temp': 20, 'max_temp': 35, 'min_rainfall': 1000, 'max_rainfall': 2000, 'optimal_ph': (6.0, 7.5)},
            'groundnut': {'min_temp': 20, 'max_temp': 30, 'min_rainfall': 500, 'max_rainfall': 1000, 'optimal_ph': (6.0, 7.0)}
        }
        
        print("✅ AgriculturalRAGPipeline initialized")
    
    def generate_recommendation(self, soil_properties, climate_conditions, user_preferences=None):
        """
        Generate evidence-backed crop recommendations
        """
        # Create query from soil and climate conditions
        query = self._create_query(soil_properties, climate_conditions)
        
        # Retrieve relevant triples
        relevant_triples = self.retriever.retrieve_relevant_triples(query, top_k=15)
        
        # Format context for LLM
        context = self._format_context(relevant_triples, soil_properties, climate_conditions)
        
        # Generate recommendation using LLM
        recommendation = self._generate_with_llm(context, user_preferences)
        
        return {
            'recommendation': recommendation,
            'evidence_triples': relevant_triples,
            'query': query,
            'context': context
        }
    
    def _create_query(self, soil_properties, climate_conditions):
        """Create a query string from soil and climate conditions"""
        query_parts = []
        
        # Add soil properties
        if 'pH' in soil_properties:
            query_parts.append(f"pH {soil_properties['pH']}")
        if 'organic_matter' in soil_properties:
            query_parts.append(f"organic matter {soil_properties['organic_matter']}")
        if 'texture_class' in soil_properties:
            query_parts.append(f"{soil_properties['texture_class']} soil")
        
        # Add climate conditions
        if 'temperature_mean' in climate_conditions:
            query_parts.append(f"temperature {climate_conditions['temperature_mean']}")
        if 'rainfall_mean' in climate_conditions:
            query_parts.append(f"rainfall {climate_conditions['rainfall_mean']}")
        
        return " ".join(query_parts)
    
    def _format_context(self, relevant_triples, soil_properties, climate_conditions):
        """Format context for LLM"""
        context_parts = []
        
        # Add soil and climate context
        context_parts.append("SOIL AND CLIMATE CONDITIONS:")
        context_parts.append(f"Soil pH: {soil_properties.get('pH', 'Unknown')}")
        context_parts.append(f"Organic Matter: {soil_properties.get('organic_matter', 'Unknown')}")
        context_parts.append(f"Soil Texture: {soil_properties.get('texture_class', 'Unknown')}")
        context_parts.append(f"Temperature: {climate_conditions.get('temperature_mean', 'Unknown')}°C")
        context_parts.append(f"Rainfall: {climate_conditions.get('rainfall_mean', 'Unknown')}mm")
        context_parts.append("")
        
        # Add relevant knowledge graph triples
        context_parts.append("RELEVANT AGRICULTURAL KNOWLEDGE:")
        for i, triple_emb in enumerate(relevant_triples[:10]):
            triple = triple_emb['triple']
            context_parts.append(f"{i+1}. {triple['subject']} {triple['predicate']} {triple['object']}")
            if 'evidence' in triple:
                context_parts.append(f"   Evidence: {triple['evidence']}")
        
        return "\n".join(context_parts)
    
    def _generate_with_llm(self, context, user_preferences=None):
        """Generate recommendation using LLM"""
        prompt = f"""
        You are an expert agricultural advisor specializing in Uganda/East Africa. Based on the provided soil and climate conditions and agricultural knowledge, provide evidence-backed crop recommendations.

        {context}

        Please provide:
        1. Top 3 most suitable crops with confidence scores
        2. Specific reasons for each recommendation based on the evidence
        3. Management practices needed for optimal yield
        4. Potential challenges and mitigation strategies
        5. Expected yield ranges

        Format your response as a structured recommendation with clear sections.
        """
        
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating recommendation: {e}"

# Initialize the RAG pipeline
print("\nInitializing RAG Pipeline...")

# Create retriever
retriever = GraphEmbeddingRetriever(
    model_path=best_model_info['model_path'],
    metadata_path='/content/drive/MyDrive/Final/data/processed/trained_models/model_metadata.json',
    triples_data=unified_triples
)

# Create RAG pipeline
rag_pipeline = AgriculturalRAGPipeline(retriever, model)

print("✅ RAG Pipeline setup complete!")
print(f"   Best model: {best_model_info['model_name']}")
print(f"   Knowledge graph: {len(unified_triples)} triples")
print(f"   Entity embeddings: {len(retriever.entity_embeddings)}")

# Test the pipeline with sample data
print("\nTesting RAG Pipeline...")

# Sample soil and climate conditions
sample_soil = {
    'pH': 6.5,
    'organic_matter': 2.5,
    'texture_class': 'loamy'
}

sample_climate = {
    'temperature_mean': 25,
    'rainfall_mean': 800,
    'humidity_mean': 70
}

# Generate recommendation
recommendation_result = rag_pipeline.generate_recommendation(sample_soil, sample_climate)

print("✅ Sample recommendation generated!")
print(f"   Retrieved {len(recommendation_result['evidence_triples'])} relevant triples")
print(f"   Query: {recommendation_result['query']}")

print("\n" + "="*70)
print("RAG PIPELINE SETUP COMPLETE")
print("="*70)
print("Next steps:")
print("1. Implement advanced retrieval strategies")
print("2. Add constrained decoding for agricultural constraints")
print("3. Create multi-objective cropping planner")
print("4. Build evaluation system")
print("5. Deploy recommendation API")
