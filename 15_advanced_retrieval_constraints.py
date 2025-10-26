"""
Phase 6, Cell 2: Advanced Retrieval Strategies and Constrained Decoding
This cell implements semantic search, hybrid retrieval, and agricultural constraint enforcement
"""

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Import Gemini API
import google.generativeai as genai

# For faster testing, set this to True to skip semantic embeddings
SKIP_SEMANTIC_EMBEDDINGS = True  # Change to False for full semantic search

print("Setting up Advanced Retrieval and Constrained Decoding...")

# Configure Gemini API
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("✅ Gemini API configured for constrained decoding")
except Exception as e:
    print(f"❌ Gemini API configuration failed: {e}")
    llm_model = None

class SemanticRetriever:
    """
    Advanced semantic retrieval system using sentence transformers
    """
    
    def __init__(self, triples_data, entity_embeddings, entity_to_id, id_to_entity):
        self.triples_data = triples_data
        self.entity_embeddings = entity_embeddings
        self.entity_to_id = entity_to_id
        self.id_to_entity = id_to_entity
        
        # Initialize sentence transformer for semantic search
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded for semantic search")
        except Exception as e:
            print(f"❌ Sentence transformer loading failed: {e}")
            self.sentence_model = None
        
        # Create TF-IDF vectorizer for hybrid retrieval
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Prepare text representations of triples
        self.triple_texts = self._create_triple_texts()
        
        # Create TF-IDF matrix
        if self.triple_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.triple_texts)
            print(f"✅ TF-IDF matrix created: {self.tfidf_matrix.shape}")
        
        # Create semantic embeddings if available (with option to skip for faster testing)
        if self.sentence_model and not SKIP_SEMANTIC_EMBEDDINGS:
            try:
                self.semantic_embeddings = self._create_semantic_embeddings()
                print(f"✅ Semantic embeddings created: {len(self.semantic_embeddings)}")
            except Exception as e:
                print(f"⚠️ Error creating semantic embeddings: {e}")
                print("   Falling back to TF-IDF only retrieval")
                self.semantic_embeddings = None
        else:
            if SKIP_SEMANTIC_EMBEDDINGS:
                print("⚠️ Skipping semantic embeddings for faster testing")
            self.semantic_embeddings = None
    
    def _create_triple_texts(self):
        """Create text representations of triples for TF-IDF"""
        texts = []
        for triple in self.triples_data:
            # Create descriptive text for each triple
            text = f"{triple['subject']} {triple['predicate']} {triple['object']}"
            if 'evidence' in triple:
                text += f" {triple['evidence']}"
            texts.append(text)
        return texts
    
    def _create_semantic_embeddings(self):
        """Create semantic embeddings for all triple texts with batching"""
        if not self.sentence_model:
            return None
        
        print("Creating semantic embeddings (this may take a few minutes)...")
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        embeddings = []
        
        for i in range(0, len(self.triple_texts), batch_size):
            batch_texts = self.triple_texts[i:i+batch_size]
            batch_embeddings = self.sentence_model.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {min(i + batch_size, len(self.triple_texts))}/{len(self.triple_texts)} triple texts")
        
        # Concatenate all embeddings
        final_embeddings = np.vstack(embeddings)
        print(f"✅ Semantic embeddings created: {final_embeddings.shape}")
        
        return final_embeddings
    
    def hybrid_retrieve(self, query, top_k=15, alpha=0.7):
        """
        Hybrid retrieval combining semantic search and TF-IDF
        alpha: weight for semantic similarity (1-alpha for TF-IDF)
        """
        results = []
        
        # TF-IDF similarity
        if hasattr(self, 'tfidf_matrix'):
            query_tfidf = self.tfidf_vectorizer.transform([query])
            tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        else:
            tfidf_similarities = np.zeros(len(self.triples_data))
        
        # Semantic similarity
        if self.sentence_model and hasattr(self, 'semantic_embeddings') and self.semantic_embeddings is not None:
            query_embedding = self.sentence_model.encode([query])
            semantic_similarities = cosine_similarity(query_embedding, self.semantic_embeddings).flatten()
        else:
            semantic_similarities = np.zeros(len(self.triples_data))
        
        # Combine scores
        # Ensure arrays are the same size
        min_size = min(len(tfidf_similarities), len(semantic_similarities), len(self.triples_data))
        tfidf_similarities = tfidf_similarities[:min_size]
        semantic_similarities = semantic_similarities[:min_size]
        
        combined_scores = alpha * semantic_similarities + (1 - alpha) * tfidf_similarities
        
        # Debug: Check array sizes
        print(f"Debug: TF-IDF similarities shape: {tfidf_similarities.shape}")
        print(f"Debug: Semantic similarities shape: {semantic_similarities.shape}")
        print(f"Debug: Combined scores shape: {combined_scores.shape}")
        print(f"Debug: Number of triples: {len(self.triples_data)}")
        print(f"Debug: Min size used: {min_size}")
        
        # Get top-k results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        for idx in top_indices:
            if idx < len(self.triples_data) and combined_scores[idx] > 0.05:  # Lowered threshold for relevance
                results.append({
                    'triple': self.triples_data[idx],
                    'score': combined_scores[idx],
                    'tfidf_score': tfidf_similarities[idx],
                    'semantic_score': semantic_similarities[idx]
                })
        
        # If no results found, return top results regardless of threshold
        if not results:
            for idx in top_indices[:top_k]:
                if idx < len(self.triples_data):
                    results.append({
                        'triple': self.triples_data[idx],
                        'score': combined_scores[idx],
                        'tfidf_score': tfidf_similarities[idx],
                        'semantic_score': semantic_similarities[idx]
                    })
        
        return results

class AgriculturalConstraintEngine:
    """
    Agricultural constraint enforcement system
    """
    
    def __init__(self):
        # Comprehensive crop constraints
        self.crop_constraints = {
            'maize': {
                'temperature': {'min': 15, 'max': 30, 'optimal': (20, 25)},
                'rainfall': {'min': 500, 'max': 1200, 'optimal': (600, 800)},
                'ph': {'min': 5.5, 'max': 7.5, 'optimal': (6.0, 7.0)},
                'soil_texture': ['loamy', 'clay_loam', 'sandy_loam'],
                'organic_matter': {'min': 1.0, 'max': 5.0, 'optimal': (2.0, 3.0)},
                'growing_season': 120,
                'water_requirement': 'moderate'
            },
            'rice': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 1000, 'max': 2000, 'optimal': (1200, 1500)},
                'ph': {'min': 5.5, 'max': 6.5, 'optimal': (6.0, 6.5)},
                'soil_texture': ['clay', 'clay_loam'],
                'organic_matter': {'min': 2.0, 'max': 6.0, 'optimal': (3.0, 4.0)},
                'growing_season': 150,
                'water_requirement': 'high'
            },
            'beans': {
                'temperature': {'min': 18, 'max': 28, 'optimal': (22, 25)},
                'rainfall': {'min': 400, 'max': 1000, 'optimal': (500, 700)},
                'ph': {'min': 6.0, 'max': 7.5, 'optimal': (6.5, 7.0)},
                'soil_texture': ['loamy', 'sandy_loam', 'clay_loam'],
                'organic_matter': {'min': 1.5, 'max': 4.0, 'optimal': (2.0, 3.0)},
                'growing_season': 90,
                'water_requirement': 'moderate'
            },
            'cassava': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 600, 'max': 1500, 'optimal': (800, 1200)},
                'ph': {'min': 5.5, 'max': 7.0, 'optimal': (6.0, 6.5)},
                'soil_texture': ['sandy', 'sandy_loam', 'loamy'],
                'organic_matter': {'min': 1.0, 'max': 4.0, 'optimal': (2.0, 3.0)},
                'growing_season': 300,
                'water_requirement': 'low'
            },
            'sweet_potato': {
                'temperature': {'min': 18, 'max': 30, 'optimal': (22, 26)},
                'rainfall': {'min': 500, 'max': 1200, 'optimal': (600, 900)},
                'ph': {'min': 5.5, 'max': 6.5, 'optimal': (6.0, 6.2)},
                'soil_texture': ['sandy', 'sandy_loam', 'loamy'],
                'organic_matter': {'min': 1.0, 'max': 3.0, 'optimal': (1.5, 2.5)},
                'growing_season': 120,
                'water_requirement': 'moderate'
            },
            'banana': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 1000, 'max': 2000, 'optimal': (1200, 1800)},
                'ph': {'min': 6.0, 'max': 7.0, 'optimal': (6.2, 6.8)},
                'soil_texture': ['loamy', 'clay_loam'],
                'organic_matter': {'min': 2.0, 'max': 5.0, 'optimal': (3.0, 4.0)},
                'growing_season': 365,
                'water_requirement': 'high'
            },
            'coffee': {
                'temperature': {'min': 15, 'max': 25, 'optimal': (18, 22)},
                'rainfall': {'min': 1200, 'max': 2000, 'optimal': (1500, 1800)},
                'ph': {'min': 6.0, 'max': 6.5, 'optimal': (6.2, 6.4)},
                'soil_texture': ['loamy', 'clay_loam'],
                'organic_matter': {'min': 2.0, 'max': 5.0, 'optimal': (3.0, 4.0)},
                'growing_season': 365,
                'water_requirement': 'moderate'
            },
            'cotton': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 500, 'max': 1200, 'optimal': (600, 800)},
                'ph': {'min': 6.0, 'max': 7.0, 'optimal': (6.2, 6.8)},
                'soil_texture': ['loamy', 'sandy_loam', 'clay_loam'],
                'organic_matter': {'min': 1.0, 'max': 3.0, 'optimal': (1.5, 2.5)},
                'growing_season': 180,
                'water_requirement': 'moderate'
            },
            'sugarcane': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 1000, 'max': 2000, 'optimal': (1200, 1500)},
                'ph': {'min': 6.0, 'max': 7.5, 'optimal': (6.5, 7.0)},
                'soil_texture': ['loamy', 'clay_loam'],
                'organic_matter': {'min': 2.0, 'max': 5.0, 'optimal': (3.0, 4.0)},
                'growing_season': 365,
                'water_requirement': 'high'
            },
            'groundnut': {
                'temperature': {'min': 20, 'max': 30, 'optimal': (24, 28)},
                'rainfall': {'min': 500, 'max': 1000, 'optimal': (600, 800)},
                'ph': {'min': 6.0, 'max': 7.0, 'optimal': (6.2, 6.8)},
                'soil_texture': ['sandy', 'sandy_loam', 'loamy'],
                'organic_matter': {'min': 1.0, 'max': 3.0, 'optimal': (1.5, 2.5)},
                'growing_season': 120,
                'water_requirement': 'low'
            }
        }
        
        print("✅ Agricultural constraint engine initialized")
    
    def evaluate_crop_suitability(self, crop, soil_properties, climate_conditions):
        """
        Evaluate crop suitability based on constraints
        Returns: (suitability_score, constraint_violations, recommendations)
        """
        if crop not in self.crop_constraints:
            return 0.0, ["Unknown crop"], []
        
        constraints = self.crop_constraints[crop]
        violations = []
        recommendations = []
        score = 1.0
        
        # Temperature evaluation
        temp = climate_conditions.get('temperature_mean', 0)
        if temp < constraints['temperature']['min'] or temp > constraints['temperature']['max']:
            violations.append(f"Temperature {temp}°C outside range {constraints['temperature']['min']}-{constraints['temperature']['max']}°C")
            score -= 0.2  # Reduced penalty
        elif temp < constraints['temperature']['optimal'][0] or temp > constraints['temperature']['optimal'][1]:
            recommendations.append(f"Temperature {temp}°C is suboptimal. Optimal range: {constraints['temperature']['optimal'][0]}-{constraints['temperature']['optimal'][1]}°C")
            score -= 0.05  # Reduced penalty
        
        # Rainfall evaluation
        rainfall = climate_conditions.get('rainfall_mean', 0)
        if rainfall < constraints['rainfall']['min'] or rainfall > constraints['rainfall']['max']:
            violations.append(f"Rainfall {rainfall}mm outside range {constraints['rainfall']['min']}-{constraints['rainfall']['max']}mm")
            score -= 0.2  # Reduced penalty
        elif rainfall < constraints['rainfall']['optimal'][0] or rainfall > constraints['rainfall']['optimal'][1]:
            recommendations.append(f"Rainfall {rainfall}mm is suboptimal. Optimal range: {constraints['rainfall']['optimal'][0]}-{constraints['rainfall']['optimal'][1]}mm")
            score -= 0.05  # Reduced penalty
        
        # pH evaluation
        ph = soil_properties.get('pH', 7.0)
        if ph < constraints['ph']['min'] or ph > constraints['ph']['max']:
            violations.append(f"pH {ph} outside range {constraints['ph']['min']}-{constraints['ph']['max']}")
            score -= 0.15  # Reduced penalty
        elif ph < constraints['ph']['optimal'][0] or ph > constraints['ph']['optimal'][1]:
            recommendations.append(f"pH {ph} is suboptimal. Optimal range: {constraints['ph']['optimal'][0]}-{constraints['ph']['optimal'][1]}")
            score -= 0.03  # Reduced penalty
        
        # Soil texture evaluation
        texture = soil_properties.get('texture_class', '').lower()
        if texture and texture not in constraints['soil_texture']:
            violations.append(f"Soil texture '{texture}' not suitable. Suitable textures: {constraints['soil_texture']}")
            score -= 0.1  # Reduced penalty
        
        # Organic matter evaluation
        om = soil_properties.get('organic_matter', 0)
        if om < constraints['organic_matter']['min'] or om > constraints['organic_matter']['max']:
            violations.append(f"Organic matter {om}% outside range {constraints['organic_matter']['min']}-{constraints['organic_matter']['max']}%")
            score -= 0.05  # Reduced penalty
        elif om < constraints['organic_matter']['optimal'][0] or om > constraints['organic_matter']['optimal'][1]:
            recommendations.append(f"Organic matter {om}% is suboptimal. Optimal range: {constraints['organic_matter']['optimal'][0]}-{constraints['organic_matter']['optimal'][1]}%")
            score -= 0.02  # Reduced penalty
        
        # Ensure score doesn't go below 0
        score = max(0.0, score)
        
        return score, violations, recommendations
    
    def get_suitable_crops(self, soil_properties, climate_conditions, min_score=0.3):
        """
        Get all crops suitable for given conditions (lowered threshold for more flexibility)
        """
        suitable_crops = []
        
        for crop in self.crop_constraints.keys():
            score, violations, recommendations = self.evaluate_crop_suitability(
                crop, soil_properties, climate_conditions
            )
            
            if score >= min_score:
                suitable_crops.append({
                    'crop': crop,
                    'suitability_score': score,
                    'violations': violations,
                    'recommendations': recommendations
                })
        
        # Sort by suitability score
        suitable_crops.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return suitable_crops

class ConstrainedRAGPipeline:
    """
    Enhanced RAG pipeline with constraint enforcement
    """
    
    def __init__(self, semantic_retriever, constraint_engine, llm_model):
        self.semantic_retriever = semantic_retriever
        self.constraint_engine = constraint_engine
        self.llm_model = llm_model
        
        print("✅ Constrained RAG Pipeline initialized")
    
    def generate_constrained_recommendation(self, soil_properties, climate_conditions, user_preferences=None):
        """
        Generate constraint-aware crop recommendations
        """
        # First, get suitable crops using constraint engine
        suitable_crops = self.constraint_engine.get_suitable_crops(soil_properties, climate_conditions)
        
        if not suitable_crops:
            return {
                'recommendation': self._generate_fallback_recommendation(soil_properties, climate_conditions),
                'suitable_crops': [],
                'constraint_analysis': "No crops meet minimum suitability requirements. Providing general agricultural advice.",
                'evidence_triples': []
            }
        
        # Create query for semantic retrieval
        query = self._create_enhanced_query(soil_properties, climate_conditions, suitable_crops)
        
        # Retrieve relevant evidence
        evidence_results = self.semantic_retriever.hybrid_retrieve(query, top_k=20)
        
        # Format context with constraints
        context = self._format_constrained_context(
            evidence_results, soil_properties, climate_conditions, suitable_crops
        )
        
        # Generate constrained recommendation
        recommendation = self._generate_constrained_llm_response(context, suitable_crops, user_preferences)
        
        return {
            'recommendation': recommendation,
            'suitable_crops': suitable_crops,
            'evidence_triples': evidence_results,
            'constraint_analysis': self._analyze_constraints(suitable_crops),
            'query': query
        }
    
    def _create_enhanced_query(self, soil_properties, climate_conditions, suitable_crops):
        """Create enhanced query including suitable crops"""
        query_parts = []
        
        # Add soil and climate conditions
        query_parts.append(f"pH {soil_properties.get('pH', 'unknown')}")
        query_parts.append(f"organic matter {soil_properties.get('organic_matter', 'unknown')}")
        query_parts.append(f"{soil_properties.get('texture_class', 'unknown')} soil")
        query_parts.append(f"temperature {climate_conditions.get('temperature_mean', 'unknown')}")
        query_parts.append(f"rainfall {climate_conditions.get('rainfall_mean', 'unknown')}")
        
        # Add suitable crops
        crop_names = [crop['crop'] for crop in suitable_crops[:5]]
        query_parts.append(f"suitable crops: {', '.join(crop_names)}")
        
        return " ".join(query_parts)
    
    def _format_constrained_context(self, evidence_results, soil_properties, climate_conditions, suitable_crops):
        """Format context with constraint information"""
        context_parts = []
        
        # Add conditions
        context_parts.append("AGRICULTURAL CONDITIONS:")
        context_parts.append(f"Soil pH: {soil_properties.get('pH', 'Unknown')}")
        context_parts.append(f"Organic Matter: {soil_properties.get('organic_matter', 'Unknown')}%")
        context_parts.append(f"Soil Texture: {soil_properties.get('texture_class', 'Unknown')}")
        context_parts.append(f"Temperature: {climate_conditions.get('temperature_mean', 'Unknown')}°C")
        context_parts.append(f"Rainfall: {climate_conditions.get('rainfall_mean', 'Unknown')}mm")
        context_parts.append("")
        
        # Add constraint analysis
        context_parts.append("CONSTRAINT-BASED CROP SUITABILITY:")
        for crop_info in suitable_crops[:5]:
            context_parts.append(f"• {crop_info['crop'].title()}: Suitability Score {crop_info['suitability_score']:.2f}")
            if crop_info['recommendations']:
                context_parts.append(f"  Recommendations: {'; '.join(crop_info['recommendations'])}")
        context_parts.append("")
        
        # Add evidence
        context_parts.append("EVIDENCE FROM KNOWLEDGE GRAPH:")
        for i, result in enumerate(evidence_results[:10]):
            triple = result['triple']
            context_parts.append(f"{i+1}. {triple['subject']} {triple['predicate']} {triple['object']}")
            context_parts.append(f"   Relevance Score: {result['score']:.3f}")
            if 'evidence' in triple:
                context_parts.append(f"   Evidence: {triple['evidence']}")
        
        return "\n".join(context_parts)
    
    def _generate_constrained_llm_response(self, context, suitable_crops, user_preferences):
        """Generate LLM response with constraint awareness"""
        prompt = f"""
        You are an expert agricultural advisor for Uganda/East Africa. Based on the constraint analysis and evidence provided, give a structured crop recommendation.

        {context}

        Please provide:
        1. TOP 3 RECOMMENDED CROPS with confidence scores (0-1)
        2. DETAILED JUSTIFICATION for each recommendation based on constraints and evidence
        3. MANAGEMENT PRACTICES needed for optimal yield
        4. RISK ASSESSMENT and mitigation strategies
        5. EXPECTED YIELD RANGES and economic considerations
        6. SEASONAL PLANNING recommendations

        Format your response with clear sections and bullet points.
        Focus on evidence-backed recommendations that respect agricultural constraints.
        """
        
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating constrained recommendation: {e}"
    
    def _analyze_constraints(self, suitable_crops):
        """Analyze constraint satisfaction"""
        if not suitable_crops:
            return "No crops meet minimum suitability requirements"
        
        high_suitability = [c for c in suitable_crops if c['suitability_score'] >= 0.8]
        medium_suitability = [c for c in suitable_crops if 0.6 <= c['suitability_score'] < 0.8]
        
        analysis = f"Constraint Analysis: {len(high_suitability)} high-suitability crops, {len(medium_suitability)} medium-suitability crops"
        
        return analysis
    
    def _generate_fallback_recommendation(self, soil_properties, climate_conditions):
        ph = soil_properties.get('pH', 7.0)
        temp = climate_conditions.get('temperature_mean', 25)
        rainfall = climate_conditions.get('rainfall_mean', 800)
        
        # General advice based on conditions
        advice_parts = []
        
        # pH advice
        if ph < 6.0:
            advice_parts.append("• Consider soil liming to raise pH to 6.0-7.0 range")
        elif ph > 7.5:
            advice_parts.append("• Consider sulfur application to lower pH")
        
        # Temperature advice
        if temp > 30:
            advice_parts.append("• High temperatures suggest drought-tolerant crops like cassava or sorghum")
        elif temp < 20:
            advice_parts.append("• Cool temperatures suggest crops like potatoes or cool-season vegetables")
        
        # Rainfall advice
        if rainfall < 500:
            advice_parts.append("• Low rainfall suggests drought-tolerant crops and irrigation planning")
        elif rainfall > 1500:
            advice_parts.append("• High rainfall suggests rice or water-loving crops")
        
        # General recommendations
        advice_parts.extend([
            "• Consider soil improvement through organic matter addition",
            "• Implement crop rotation to improve soil health",
            "• Consult local agricultural extension services for specific advice",
            "• Consider mixed cropping systems for risk diversification"
        ])
        
        return f"""
FALLBACK AGRICULTURAL RECOMMENDATIONS

Current Conditions Analysis:
- Soil pH: {ph}
- Temperature: {temp}°C  
- Rainfall: {rainfall}mm

General Recommendations:
{chr(10).join(advice_parts)}

Note: No crops met the minimum suitability requirements. Consider soil amendments and management practices to improve conditions for crop production.
        """

# Initialize advanced retrieval and constraint systems
print("\nInitializing Advanced Retrieval and Constraint Systems...")

# Load existing data
try:
    with open('/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph.json', 'r') as f:
        triples_data = json.load(f)
    
    with open('/content/drive/MyDrive/Final/data/processed/trained_models/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Load the best model for entity embeddings
    with open('/content/drive/MyDrive/Final/data/processed/trained_models/best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    print(f"✅ Data loaded: {len(triples_data)} triples, {model_metadata['num_entities']} entities")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")

# Create semantic retriever
semantic_retriever = SemanticRetriever(
    triples_data=triples_data,
    entity_embeddings=None,  # Will be loaded from model if needed
    entity_to_id=model_metadata['entity_to_id'],
    id_to_entity=model_metadata['id_to_entity']
)

# Create constraint engine
constraint_engine = AgriculturalConstraintEngine()

# Create constrained RAG pipeline
constrained_rag = ConstrainedRAGPipeline(
    semantic_retriever=semantic_retriever,
    constraint_engine=constraint_engine,
    llm_model=llm_model
)

print("✅ Advanced retrieval and constraint systems initialized!")

# Test the enhanced system
print("\nTesting Enhanced RAG Pipeline...")

# Test with more realistic conditions
test_soil = {
    'pH': 6.2,
    'organic_matter': 2.1,
    'texture_class': 'loamy'
}

test_climate = {
    'temperature_mean': 24,
    'rainfall_mean': 750,
    'humidity_mean': 65
}

# Generate constrained recommendation
result = constrained_rag.generate_constrained_recommendation(test_soil, test_climate)

print("✅ Constrained recommendation generated!")
print(f"   Suitable crops found: {len(result['suitable_crops'])}")
print(f"   Evidence triples retrieved: {len(result['evidence_triples'])}")
print(f"   Constraint analysis: {result['constraint_analysis']}")

print("\n" + "="*70)
print("ADVANCED RETRIEVAL AND CONSTRAINED DECODING COMPLETE")
print("="*70)
print("Next steps:")
print("1. Implement multi-objective cropping planner (MILP/CP-SAT)")
print("2. Build comprehensive evaluation system")
print("3. Create recommendation API")
print("4. Deploy production system")
print("5. Performance optimization and monitoring")
