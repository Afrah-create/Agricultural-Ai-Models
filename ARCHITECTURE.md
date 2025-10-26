# Agricultural Recommendation System - Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [System Components](#system-components)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Machine Learning Models](#machine-learning-models)
7. [API Endpoints](#api-endpoints)
8. [Deployment Architecture](#deployment-architecture)

---

## System Overview

The Agricultural Recommendation System is an AI-powered decision support platform designed for crop recommendations in Uganda. It combines multiple AI approaches including knowledge graphs, graph neural networks (GNN), constraint-based reasoning, Retrieval-Augmented Generation (RAG), and Large Language Models (LLM) to provide evidence-based agricultural recommendations.

### Key Features
- **Multi-AI Integration**: Combines constraint-based reasoning, knowledge graphs, GNN embeddings, RAG, and LLM
- **Evidence-Based Recommendations**: Uses real Ugandan agricultural data and scientific literature
- **Multi-Factor Analysis**: Considers soil properties, climate conditions, and farming practices
- **Automated Reporting**: Generates comprehensive PDF reports
- **Scalable Deployment**: Docker-based containerization with cloud deployment support

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                           │
│          (Flask Web Application - HTML/CSS/JS)              │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (Flask)                         │
│  • /api/recommend (POST)                                     │
│  • /api/download_pdf (POST)                                  │
│  • / (Home)                                                   │
└────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Agricultural API Core Engine                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Agricultural Constraint Engine                        │   │
│  │  • Crop suitability evaluation                        │   │
│  │  • Multi-factor constraint checking                   │   │
│  │  • pH, nutrients, climate validation                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Semantic Retriever (RAG Pipeline)                    │   │
│  │  • Hybrid TF-IDF retrieval                           │   │
│  │  • Knowledge graph search                            │   │
│  │  • Evidence-based recommendations                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LLM Integration Layer                                 │   │
│  │  • Fine-tuned DialoGPT model                          │   │
│  │  • Google Gemini API (fallback)                        │   │
│  │  • Expert analysis generation                        │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data & Model Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Knowledge   │  │   Trained    │  │  Ugandan    │        │
│  │   Graph      │  │   Models     │  │   Dataset   │        │
│  │   (175K+     │  │  (GCN,       │  │  (Soil &    │        │
│  │   triples)   │  │   TransE,    │  │   Climate)  │        │
│  │              │  │   etc.)      │  │             │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │  Literature  │  │   Fine-tuned │                           │
│  │   Triples    │  │     LLM     │                           │
│  │              │  │  (DialoGPT) │                           │
│  └──────────────┘  └──────────────┘                           │
└────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. **Agricultural Constraint Engine**
**Purpose**: Rule-based validation of crop suitability based on agricultural constraints

**Location**: `deployment/app/main.py` (lines 277-454)

**Key Features**:
- Evaluates crop suitability against soil properties, climate conditions, and nutrient requirements
- Implements constraints for 8+ major crops (maize, rice, beans, cassava, etc.)
- Validates: pH range, organic matter, soil texture, temperature, rainfall, NPK levels
- Generates recommendations for improving soil conditions
- Calculates suitability scores (0-1)

**Crop Constraints**:
```python
{
    'maize': {
        'pH_range': (5.5, 7.5),
        'organic_matter_min': 1.0,
        'temperature_range': (18, 30),
        'rainfall_range': (500, 1500),
        'soil_textures': ['loam', 'clay_loam', 'sandy_loam'],
        'nitrogen_range': (50, 200),
        'phosphorus_range': (10, 50),
        'potassium_range': (80, 300)
    },
    # ... similar for other crops
}
```

---

### 2. **Semantic Retriever (RAG Pipeline)**
**Purpose**: Hybrid retrieval system for finding relevant agricultural knowledge from knowledge graph

**Location**: `deployment/app/main.py` (lines 205-276)

**Technology**: TF-IDF + Cosine Similarity

**Process**:
1. Creates text representations of all triples: `"{subject} {predicate} {object} {evidence}"`
2. Builds TF-IDF vectorizer with ngram_range=(1, 2), max_features=1000
3. Computes cosine similarity between query and triple texts
4. Returns top-k most relevant triples as evidence

**Key Method**: `hybrid_retrieve(query, top_k=15)`

---

### 3. **Fine-Tuned LLM Component**
**Purpose**: Domain-specific LLM for generating agricultural recommendations

**Location**: `deployment/app/main.py` (lines 455-512)

**Model**: DialoGPT-small (Microsoft) - Conversational model optimized for dialogue

**Training Data**:
- Agricultural knowledge graph triples
- Literature review data
- Ugandan agricultural dataset
- Domain-specific prompts

**Capabilities**:
- Generates contextual recommendations
- Understands agricultural terminology
- Provides expert-like insights

---

### 4. **Knowledge Graph**
**Purpose**: Structured representation of agricultural domain knowledge

**Location**: `deployment/processed/unified_knowledge_graph.json`

**Statistics**:
- **Total Triples**: 175,318
- **Node Types**: Crops, Soil Properties, Agro-Ecological Zones, Management Practices, Climate Zones
- **Relationship Types**: has_nutrient_requirement, requires_ph, prefers_texture_class, suitability, recommended_for, requires_climate

**Schema Structure** (`unified_knowledge_graph_schema.json`):
```json
{
  "node_types": {
    "crop": {
      "attributes": ["name", "scientific_name", "family", "growth_period"]
    },
    "soil_property": {
      "attributes": ["name", "unit", "optimal_range"]
    },
    "agro_ecological_zone": {
      "attributes": ["zone_id", "description", "climate_characteristics"]
    }
  },
  "relationship_types": {
    "has_nutrient_requirement": {...},
    "requires_ph": {...},
    "prefers_texture_class": {...}
  }
}
```

**Data Sources**:
- **Dataset Triples**: Extracted from Ugandan agricultural dataset
- **Literature Triples**: Extracted from 52 PDF literature reviews using NLP

---

### 5. **Graph Neural Network Models**
**Purpose**: Learn embeddings for entities and relations in the knowledge graph

**Location**: `deployment/processed/trained_models/`

**Models Implemented**:
1. **TransE** (`transe_model.pth`) - Translation-based embedding
2. **DistMult** (`distmult_model.pth`) - Bilinear diagonal model
3. **ComplEx** (`complex_model.pth`) - Complex embeddings
4. **GCN** (`gcn_model.pth`) - Graph Convolutional Network
5. **GraphSAGE** (`graphsage_model.pth`) - GraphSAGE architecture

**Best Performing Model**: GCN
- Accuracy: 87.28%
- Precision: 97.70%
- Recall: 76.35%
- F1-Score: 85.71%
- ROC-AUC: 96.90%

**Model Metadata**:
- Embedding Dimension: 100
- Hidden Dimension: 200 (for GCN/GraphSAGE)
- Entities: 5,069
- Relations: 194

---

### 6. **Data Loader Component**
**Purpose**: Manages loading and caching of all data assets

**Location**: `deployment/app/main.py` (lines 115-166)

**Data Assets Loaded**:
1. **Unified Knowledge Graph**: 175K+ triples
2. **Dataset Triples**: Extracted from Ugandan CSV data
3. **Literature Triples**: Extracted from PDF literature
4. **Ugandan Dataset**: Cleaned CSV with soil/climate data

**Initialization Flow**:
```python
def load_data(self):
    # Load unified knowledge graph
    kg_path = "processed/unified_knowledge_graph.json"
    # Load dataset triples
    dataset_triples_path = "processed/dataset_triples.json"
    # Load literature triples
    literature_triples_path = "processed/literature_triples.json"
    # Load Ugandan dataset
    ugandan_data_path = "processed/ugandan_data_cleaned.csv"
```

---

### 7. **Agricultural API Main Engine**
**Purpose**: Orchestrates all components and generates recommendations

**Location**: `deployment/app/main.py` (lines 514-1346)

**Key Methods**:

**a) `get_recommendation()`**:
- Evaluates all crops against constraints
- Filters crops based on farming conditions
- Generates land allocation (if available land provided)
- Calculates evaluation scores (economic, environmental, social, risk)
- Creates unified recommendation text combining all AI sources

**b) `_generate_recommendation_text()`**:
- Creates structured recommendation with sections:
  - Executive summary
  - AI analysis (fine-tuned LLM)
  - Expert analysis (Gemini API)
  - Technical analysis
  - Implementation plan

**c) `_get_structured_recommendation_sections()`**:
- Returns recommendation in structured format for better UI display
- Separates each analysis into distinct sections:
  - Primary recommendation with crop name and score
  - AI analysis section
  - Expert analysis section
  - Technical analysis section
  - Implementation plan section
- Used for improved visual presentation in web interface

**c) `generate_pdf_report()`**:
- Creates professional PDF using ReportLab
- Includes: title page, soil analysis, crop recommendations, implementation plan
- Returns bytes for download

**d) `_generate_land_allocation()`**:
- Allocates land proportionally based on suitability scores
- Returns crop-wise land allocation plan

---

## Data Flow

### Request Flow:
```
1. User submits soil properties + climate conditions via web form
   ↓
2. API validates required fields:
   - Soil: pH, organic_matter, texture_class, nitrogen, phosphorus, potassium
   - Climate: temperature_mean, rainfall_mean
   ↓
3. AgriculturalAPI.get_recommendation() called
   ↓
4. For each available crop:
   - AgriculturalConstraintEngine.evaluate_crop_suitability()
   - Check pH, OM, texture, temperature, rainfall, NPK
   - Calculate suitability score
   ↓
5. Filter crops based on:
   - Suitable = True
   - Farming conditions (e.g., organic farming, irrigation availability)
   ↓
6. Sort crops by suitability score
   ↓
7. Generate additional insights:
   - Land allocation (if available land provided)
   - Evaluation scores (economic, environmental, social, risk)
   ↓
8. Generate unified recommendation text:
   - Fine-tuned LLM insights
   - Gemini API expert analysis
   - Technical analysis
   - Implementation plan
   ↓
9. Return structured JSON response
```

### RAG Retrieval Flow:
```
1. User query formed from: soil properties + climate conditions + suitable crops
   ↓
2. SemanticRetriever.hybrid_retrieve(query, top_k=15)
   ↓
3. Query transformed to TF-IDF vector
   ↓
4. Compute cosine similarity with all triple texts
   ↓
5. Return top-k relevant triples as evidence
   ↓
6. Triples used as context in LLM generation
```

---

## Technology Stack

### Frontend
- **HTML5/CSS3/JavaScript**: Modern web interface
- **Responsive Design**: Mobile-friendly UI
- **Client-side validation**: Form validation before API calls

### Backend
- **Flask 3.1.2**: Web framework
- **Flask-CORS 6.0.1**: Cross-origin resource sharing
- **Python 3.11**: Programming language

### Machine Learning
- **PyTorch 2.0.1**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **scikit-learn 1.3.0**: TF-IDF vectorization and similarity
- **NumPy 1.24.3**: Numerical computations

### AI/LLM
- **Google Gemini API**: Expert analysis generation
- **Fine-tuned DialoGPT**: Domain-specific LLM
- **sentence-transformers**: Embedding models

### Data Processing
- **Pandas 2.0.3**: Data manipulation
- **NetworkX**: Knowledge graph construction

### Reporting
- **ReportLab 4.0.4**: PDF generation

### Optimization
- **OR-Tools 9.7.2996**: Constraint optimization

### Deployment
- **Docker**: Containerization
- **Gunicorn 21.2.0**: WSGI HTTP server
- **python-dotenv**: Environment variable management

---

## Machine Learning Models

### 1. Graph Embedding Models

#### TransE
- **Type**: Translational distance model
- **Principle**: Entities and relations represented in same space
- **Objective**: Minimize distance between (h + r) and t
- **Use Case**: Basic knowledge graph link prediction

#### DistMult
- **Type**: Bilinear model with diagonal matrices
- **Principle**: Efficient parameter sharing
- **Advantage**: Faster training than TransE
- **Use Case**: Dense knowledge graph embeddings

#### ComplEx
- **Type**: Complex embeddings
- **Principle**: Handles asymmetric relations
- **Advantage**: Better for non-symmetric relations
- **Use Case**: Relations with directionality

#### GCN (Graph Convolutional Network)
- **Type**: GNN with message passing
- **Architecture**: 
  - Embedding layer: 100 dim
  - Hidden layer: 200 dim
  - Graph convolution
  - Output layer
- **Performance** (Best Model):
  - Accuracy: 87.28%
  - Precision: 97.70%
  - Recall: 76.35%
  - F1-Score: 85.71%
  - ROC-AUC: 96.90%
- **Use Case**: Best overall performance for entity classification

#### GraphSAGE
- **Type**: Inductive GNN with sampling
- **Architecture**: Neighborhood sampling + aggregation
- **Advantage**: Can handle new nodes not seen in training
- **Use Case**: Dynamic knowledge graphs

### 2. Fine-Tuned LLM

**Base Model**: DistilGPT-2 (Microsoft)
- **Parameters**: ~117M
- **Architecture**: GPT-2 based dialogue model
- **Fine-tuning**: Agricultural domain data
- **Features**:
  - Conversational response generation
  - Domain-specific agricultural terminology
  - Context-aware recommendations

**Training Data Sources**:
- Knowledge graph triples converted to text
- Literature review extracts
- Agricultural fact statements
- Crop recommendation examples

---

## API Endpoints

### 1. **POST /api/recommend**
**Purpose**: Get crop recommendations

**Request Body**:
```json
{
  "soil_properties": {
    "pH": 6.5,
    "organic_matter": 2.1,
    "texture_class": "loam",
    "nitrogen": 120,
    "phosphorus": 35,
    "potassium": 180
  },
  "climate_conditions": {
    "temperature_mean": 24,
    "rainfall_mean": 1200
  },
  "farming_conditions": {
    "available_land": 5.0,
    "organic_farming": false,
    "irrigation_available": true
  }
}
```

**Response**:
```json
{
  "suitable_crops": [
    {
      "crop": "maize",
      "suitability_score": 0.95,
      "recommendations": ["Add lime if pH drops"],
      "violations": []
    }
  ],
  "land_allocation": {
    "total_land_used": 5.0,
    "crop_details": [...]
  },
  "evaluation_scores": {
    "overall_score": 0.85,
    "dimension_scores": {...}
  },
  "recommendation_text": "...",
  "data_sources": {...}
}
```

**Validation Rules**:
- Required fields: pH, organic_matter, texture_class, temperature_mean, rainfall_mean
- Optional fields: nitrogen, phosphorus, potassium, available_land

---

### 2. **POST /api/download_pdf**
**Purpose**: Download recommendation as PDF

**Request Body**: Same as `/api/recommend`

**Response**: PDF file download

---

### 3. **GET /**
**Purpose**: Render web interface

**Response**: HTML page with form for inputting soil/climate data

---

## Deployment Architecture

### Local Deployment
```bash
cd deployment
pip install -r requirements.txt
python app/main.py
```

**Access**: http://localhost:5000

### Docker Deployment
```bash
cd deployment
docker build -t agricultural-api .
docker run -p 5000:5000 agricultural-api
```

**Dockerfile Structure**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app/main.py"]
```

### Cloud Deployment Options

#### 1. AWS EC2
- Launch Ubuntu 20.04 instance
- Install Docker
- Deploy application

#### 2. Google Cloud Run
- Containerized deployment
- Automatic scaling
- Pay-per-use pricing

#### 3. Azure Container Instances
- Container-based deployment
- Resource group management
- Azure networking

---

## System Workflow

### Initialization Flow:
```
1. Application starts (main.py)
   ↓
2. Configure Gemini API (if GEMINI_API_KEY available)
   ↓
3. Initialize AgriculturalAPI:
   - DataLoader: Load knowledge graph, triples, dataset
   - ModelLoader: Load GCN model + metadata
   - ConstraintEngine: Initialize crop constraints
   - SemanticRetriever: Initialize TF-IDF (if RAG available)
   - FineTunedLLM: Load fine-tuned model
   ↓
4. All components loaded and ready
   ↓
5. Flask app starts listening on port 5000
```

### Recommendation Generation Flow:
```
1. User submits form → POST /api/recommend
   ↓
2. Validate input (soil + climate data)
   ↓
3. AgriculturalAPI.get_recommendation():
   a) Evaluate all crops (ConstraintEngine)
   b) Filter suitable crops
   c) Calculate land allocation (if available)
   d) Generate evaluation scores
   e) Create recommendation text (combines all AI sources):
      - AI insights from fine-tuned LLM
      - Expert analysis from Gemini
      - Technical analysis
      - Implementation plan
   ↓
4. Return JSON response with:
   - Suitable crops (with scores)
   - Land allocation plan
   - Evaluation scores
   - Recommendation text
   - Data sources
```

---

## Key Design Decisions

### 1. **Hybrid AI Approach**
Combines multiple AI paradigms:
- **Constraint-based reasoning**: Rule-based validation
- **Graph embeddings**: Deep learning on knowledge graph
- **RAG**: Retrieval for evidence-based recommendations
- **LLM**: Natural language understanding and generation

**Rationale**: No single AI approach perfect → ensemble approach provides robust recommendations

### 2. **Multi-Layer Fallback System**
```
Primary: Fine-tuned LLM
  ↓ (fallback if unavailable)
Gemini API
  ↓ (fallback if unavailable)
Template-based expert analysis
```

**Rationale**: Ensures system always provides recommendations even if AI services unavailable

### 3. **TF-IDF for RAG**
**Choice**: Traditional TF-IDF instead of embeddings
- **Reason**: Simpler, faster, interpretable
- **Trade-off**: Slightly less accurate than dense embeddings
- **Benefit**: Lower latency for production

### 4. **GCN as Best Model**
**Choice**: GCN selected over TransE, DistMult, ComplEx, GraphSAGE
- **Reason**: Best F1-score (85.71%), ROC-AUC (96.90%)
- **Architecture**: Captures graph structure through convolution
- **Use**: Entity classification and link prediction

### 5. **Flask Over FastAPI/Django**
**Choice**: Flask for lightweight API
- **Reason**: Simpler for MVP, easier deployment
- **Trade-off**: Less built-in features than Django
- **Benefit**: Full control over architecture

---

## Data Sources

### 1. **Ugandan Agricultural Dataset**
**Format**: CSV
**Location**: `Dataset/Ugandan_data.csv`
**Fields**: Soil properties, climate conditions, crop records

### 2. **Literature Reviews**
**Format**: PDF (52 files)
**Location**: `Literature_reviews/*.pdf`
**Processing**: NLP extraction of triples (subject-predicate-object)

### 3. **Knowledge Graph**
**Format**: JSON (unified triples)
**Location**: `deployment/processed/unified_knowledge_graph.json`
**Size**: 175,318 triples
**Created from**: Dataset + Literature triples

---

## Scalability Considerations

### Current Limitations:
- Single-threaded Flask app (development mode)
- No caching layer for TF-IDF matrix
- No database (all data loaded in memory)
- No distributed inference

### Scaling Strategies:

1. **Add Gunicorn Workers**:
   ```python
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Add Caching**:
   - Cache TF-IDF matrix in memory
   - Cache model inferences using Redis

3. **Add Database**:
   - MongoDB for knowledge graph storage
   - PostgreSQL for user queries/history

4. **Model Optimization**:
   - Quantize models (FP16/INT8)
   - Use ONNX for faster inference
   - Implement batch processing

5. **Load Balancing**:
   - Deploy multiple instances
   - Use nginx for load balancing

---

## Security Considerations

### Current State:
- Input validation for API endpoints
- No authentication/authorization (public API)
- API key management for Gemini (environment variable)

### Recommended Enhancements:
1. **Add Authentication**: JWT tokens for API access
2. **Rate Limiting**: Prevent abuse (Flask-Limiter)
3. **Input Sanitization**: Validate and sanitize all inputs
4. **HTTPS**: TLS encryption in production
5. **API Key Rotation**: Regular key updates

---

## Performance Metrics

### Model Performance:
- **GCN Model**: 87.28% accuracy, 96.90% ROC-AUC
- **Best Crop Prediction**: F1-score 85.71%
- **Suitability Score**: Range 0-1, normalized across 8 constraints

### System Performance:
- **API Response Time**: < 2 seconds
- **PDF Generation**: < 3 seconds
- **Model Loading**: ~10-15 seconds (cold start)

---

## Future Enhancements

1. **Mobile App**: React Native or Flutter
2. **Soil Image Analysis**: CNN for soil texture classification
3. **Weather Integration**: Real-time weather API
4. **Yield Prediction**: Time series forecasting
5. **Pest/Disease Detection**: Image classification
6. **Market Price Analysis**: Economic recommendation
7. **Multi-lingual Support**: Local languages
8. **Offline Mode**: Edge deployment

---

## Conclusion

This Agricultural Recommendation System demonstrates a sophisticated integration of multiple AI paradigms to solve a real-world agricultural decision-making problem. The architecture is designed for scalability, maintainability, and production deployment while maintaining high accuracy and interpretability.

**Key Strengths**:
- Multi-AI ensemble approach
- Evidence-based recommendations
- Comprehensive constraint validation
- User-friendly web interface
- Professional PDF reporting
- Production-ready deployment

**Technologies Used**:
- Python, Flask, PyTorch
- Graph Neural Networks
- Large Language Models
- Knowledge Graphs
- TF-IDF Retrieval
- Docker

**Application Domain**: Agriculture, specifically crop recommendations for Uganda, adaptable to other regions with similar agricultural datasets.

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Maintainer**: System Architecture Team  
**Contact**: [Refer to project README]

