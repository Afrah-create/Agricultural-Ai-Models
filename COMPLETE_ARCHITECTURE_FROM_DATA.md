# Complete Agricultural AI System Architecture
## Starting from Dataset Creation to Production

### STAGE 0: RAW DATA COLLECTION (Source Layer)

The system begins with two primary data sources collected for Ugandan agriculture:

**Data Source 1: Ugandan Agricultural Dataset**
- Location: `Dataset/Ugandan_data.csv`
- Contents: Real agricultural data from Uganda with soil properties, climate conditions, and crop records
- Fields include: crop_name, agro_ecological_zone, pH, organic_matter, nitrogen, phosphorus, potassium, texture_class, temperature, rainfall, suitability scores, yield data
- Volume: Thousands of crop-soil-climate combinations

**Data Source 2: Literature Reviews**
- Location: `Literature_reviews/` folder
- Contents: 52 PDF files containing agricultural research papers
- Subjects: Soil science, crop suitability, climate adaptation, best practices
- Processing: Text extraction from PDFs, NLP analysis, structured data extraction

### STAGE 1: DATA PREPROCESSING (Colab Notebook Cells 01-07)

This stage transforms raw data into clean, structured formats ready for knowledge graph construction.

**Cell 01: Environment Setup** (Setup Layer)
- Installs packages: PyTorch, Transformers, NetworkX, pandas, scikit-learn, Gemini API
- Sets up Python environment for data processing and ML

**Cell 02: Import Libraries** (Import Layer)
- Imports all required libraries for data processing, ML, NLP, and visualization
- Configures pandas, matplotlib, and plotting options

**Cell 03: Project Structure Creation**
- Creates directory structure:
  - `data/processed/` - cleaned datasets
  - `data/literature/` - PDF text extracts
  - `models/` - trained model weights
  - `notebooks/` - analysis notebooks

**Cell 04: Dataset Preprocessing** (Preprocessing Engine)
- Loads Ugandan_data.csv from Google Drive
- Performs data cleaning: handles missing values, normalizes crop names (e.g., "Maize" → "maize")
- Creates derived features:
  * suitability_score - weighted combination of suitability levels
  * yield_score - normalized yield potential  
  * soil_quality_index - calculated from pH and organic matter
  * climate_suitability - based on temperature and rainfall
  * overall_suitability - combined multi-factor score
- Validates data quality and saves to: `ugandan_data_preprocessed.csv`

**Cell 05: Advanced Data Cleaning**
- Handles duplicates, outliers, data type conversions
- Creates crop mapping dictionaries
- Generates cleaning report with statistics

**Cell 06: PDF Triple Extraction**
- Extracts text from literature PDFs using pdfplumber and PyPDF2
- Uses Gemini API to extract structured triples from academic text
- Prompt: "Extract soil-crop suitability relationships"
- Output: Structured triples in format (subject, predicate, object, evidence)
- Saves to: `literature_triples.json`

**Cell 07: Complete PDF Extraction** (AI Extraction Layer)
- Processes all 52 PDF files in batch
- Extracts: soil properties, crop recommendations, management practices, nutrient requirements
- Uses AI to structure unstructured text into knowledge triples
- Generates metadata for each PDF with quality metrics

### STAGE 2: KNOWLEDGE GRAPH CONSTRUCTION (Cells 08-11)

Transforms cleaned data into knowledge graphs for AI processing.

**Cell 08: Dataset to Triples** (Triple Generation Engine)
- Converts CSV records to RDF-style triples
- Creates URIs for all entities: crops, soils, zones, nutrients
- Generates triples like:
  * Crop → has_nutrient_requirement → Nitrogen Level
  * Crop → grows_in → Agro-Ecological Zone
  * Crop → requires_ph → pH Value
  * Crop → prefers_texture → Texture Class
- Output: `dataset_triples.json` with ~50,000 triples
- Each triple contains: subject, predicate, object, value, unit, source, triple_type

**Cell 09: Comprehensive EDA** (Analysis Layer)
- Exploratory data analysis on dataset
- Generates statistics: crop distribution, soil property ranges, zone analysis
- Creates visualizations: crop frequency, pH distribution, nutrient ranges
- Saves analysis: `dataset_summary.json`

**Cell 10: Triple Integration** (Integration Engine)
- Combines dataset triples + literature triples
- Deduplicates overlapping knowledge
- Merges complementary information from different sources
- Creates unified vocabulary across all sources
- Output: `unified_knowledge_graph.json` with ~175,318 triples

**Cell 11: Knowledge Graph Construction & Visualization**
- Builds NetworkX graph from triples
- Calculates graph metrics: nodes, edges, density, centrality
- Creates visualizations of crop-soil relationships
- Generates schema: `unified_knowledge_graph_schema.json`
- Schema defines: entity types, relationship types, attribute ranges

### STAGE 3: MODEL TRAINING (Cells 12-25)

Trains machine learning models on the knowledge graph.

**Cell 12: Graph Embeddings Training** (Model Training Engine)
- Implements 5 graph embedding models:
  - TransE - Translation-based embeddings
  - DistMult - Bilinear diagonal model
  - ComplEx - Complex embeddings for asymmetric relations
  - GCN - Graph Convolutional Network (best performer)
  - GraphSAGE - Neighborhood sampling model
- Training pipeline:
  * Loads unified knowledge graph
  * Creates entity/relation ID mappings (2513 entities, 15 relations)
  * Splits into train/val/test (80/10/10)
  * Generates negative samples for training
  * Trains with Adam optimizer, early stopping
- Model architectures:
  * Embedding dimension: 100
  * GCN: 100 → 200 → 100 (hidden dim: 200)
- Evaluation metrics calculated for each model
- Best model (GCN) selected and saved
- Outputs:
  * `best_model.pth` - Best model weights
  * `gcn_model.pth`, `transe_model.pth`, etc. - All model variants
  * `model_metadata.json` - Entity/relation mappings
  * `graph_embedding_results.json` - Performance metrics

**Cell 13: Embedding Visualization**
- Visualizes learned embeddings using t-SNE
- Shows clusters of similar entities
- Validates model learning quality

**Cells 14-25: RAG and LLM Setup** (Advanced AI Components)
- **Cell 14**: Sets up RAG pipeline with TF-IDF retrieval
- **Cell 15**: Implements constraint-based recommendation filtering
- **Cells 16-23**: LLM fine-tuning pipeline:
  * Prepares agricultural training dataset
  * Fine-tunes DialoGPT-small model on agricultural text
  * Trains for domain-specific language generation
  * Outputs: `quick_fine_tuned_fast/` directory with model checkpoints
- **Cells 24-27**: Model validation and integration testing

### STAGE 4: PRODUCTION DEPLOYMENT (Railway/Cloud)

The trained models and processed data are packaged into a production API.

**Architecture Overview (5-Layer Stack)**

**Layer 1: Data Foundation**
- **Location**: `deployment/data/` and `deployment/processed/`
- **Contents**:
  - `unified_knowledge_graph.json` (175K triples) - Source of truth
  - `ugandan_data_cleaned.csv` - Reference dataset
  - `literature_triples.json` - Literature knowledge
  - `trained_models/` - GCN weights, metadata, embeddings
- **Purpose**: Persistent storage for all AI components

**Layer 2: AI Model Components**
Four specialized AI engines working in parallel:

**A. Constraint Engine** (Rule-Based AI)
- **Location**: `AgriculturalConstraintEngine` class (lines 358-535)
- **Purpose**: Fast, explainable rule validation
- **Process**:
  * Loads pre-defined constraints for 8 crops
  * For each crop, checks: pH range, temperature range, rainfall range, NPK thresholds, texture preferences
  * Calculates suitability score (0-1) based on constraint violations
  * Generates specific improvement recommendations
- **Output**: List of suitable crops with scores and violations
- **Speed**: Millisecond-level evaluation

**B. GCN Enhancement Engine** (Deep Learning AI)
- **Location**: `AgriculturalModelLoader` class (lines 221-285)
- **Purpose**: Enhances scores using learned graph embeddings
- **Process**:
  * Loads trained GCN model (100-dim embeddings)
  * Retrieves entity ID for each crop from metadata
  * Passes entity through GCN layers (100 → 200 → 100)
  * Extracts embedding vector and computes similarity
  * Boosts score for crops with strong graph associations
- **Output**: Enhanced suitability scores for each crop
- **Speed**: Sub-second per request
- **Training Data**: 175K triples from knowledge graph

**C. RAG Retrieval Engine** (Information Retrieval AI)
- **Location**: `SemanticRetriever` class (lines 286-357)
- **Purpose**: Finds relevant evidence from knowledge graph
- **Process**:
  * Builds TF-IDF vectorizer over triple texts
  * User query becomes TF-IDF vector
  * Computes cosine similarity with all triple texts
  * Retrieves top-15 most relevant triples as evidence
  * Passes evidence to LLM for context-aware generation
- **Output**: Relevant triples with similarity scores
- **Data Source**: Unified knowledge graph (175K triples)

**D. LLM Generation Engine** (Natural Language AI)
- **Location**: `FineTunedLLM` class (lines 536-626)
- **Purpose**: Generates natural language insights
- **Components**:
  * Fine-tuned DialoGPT-small (117M parameters)
  * Google Gemini API integration (fallback)
  * Template-based generator (final fallback)
- **Process**:
  * Receives suitable crops + RAG context
  * Generates: AI insights, expert analysis, recommendations
  * Uses domain-specific agricultural terminology
- **Output**: Paragraphs of expert-level advice

**Layer 3: Core Orchestration**
- **Location**: `AgriculturalAPI` class (lines 627-1887)
- **Purpose**: Coordinates all AI engines
- **Process**:
  1. Receives request with soil/climate data
  2. Calls Constraint Engine → initial scores
  3. Calls GCN Enhancement → boosted scores
  4. Calls RAG Retrieval → relevant evidence
  5. Calls LLM Generator → natural language insights
  6. Combines all outputs
  7. Generates: land allocation, evaluation scores, PDF reports
- **Output**: Complete recommendation object with all analyses

**Layer 4: API Gateway**
- **Location**: Flask routes (lines 1888-3040)
- **Endpoints**:
  - `GET /` - Web interface HTML
  - `POST /api/recommend` - JSON recommendation API
  - `POST /api/download_pdf` - PDF report download
- **Process**: HTTP handling, input validation, response formatting

**Layer 5: User Interface**
- **Technology**: HTML/CSS/JavaScript
- **Features**: Form inputs, result display, PDF download
- **Deployment**: Railway cloud platform

### DATA FLOW DIAGRAM

```
RAW DATA COLLECTION
├─ Ugandan Dataset (CSV)
└─ Literature PDFs (52 files)
    ↓
PREPROCESSING STAGE
├─ Cell 04: Data Cleaning & Feature Engineering
├─ Cell 06: PDF Text Extraction with AI
└─ Cell 07: Structured Triple Extraction
    ↓
KNOWLEDGE GRAPH STAGE  
├─ Cell 08: Dataset → Triples (50K triples)
├─ Cell 10: Integration (create 175K unified graph)
└─ Cell 11: Graph Construction & Schema
    ↓
MODEL TRAINING STAGE
├─ Cell 12: Train 5 Models (GCN selected as best)
├─ Cell 14-15: Set up RAG Pipeline
└─ Cell 16-25: Fine-tune LLM (DialoGPT)
    ↓
DEPLOYMENT STAGE
├─ Copy models to deployment/processed/trained_models/
├─ Copy knowledge graph to deployment/data/
├─ Initialize Flask API
└─ Deploy to Railway
    ↓
PRODUCTION FLOW (Per Request)
├─ User Input → API Gateway
├─ Constraint Engine → Fast Filtering
├─ GCN Enhancement → Deep Learning Boost  
├─ RAG Retrieval → Evidence Gathering
├─ LLM Generation → Expert Analysis
└─ Combined Response → User Output
```

### KEY NUMBERS

**Raw Data**:
- Dataset: Thousands of crop-soil combinations
- PDFs: 52 literature review files

**Knowledge Graph**:
- Total Triples: 175,318
- Entities: 2,513 (crops, soils, zones, practices)
- Relations: 15 types
- Source Triples: ~50K from dataset, ~125K from literature

**Models**:
- GCN (Best): 87.28% accuracy, 96.90% ROC-AUC
- Embedding Dimension: 100
- Hidden Dimension: 200
- Model Parameters: ~1M total

**Production Performance**:
- API Response Time: <2 seconds
- Supported Crops: 8 (maize, rice, beans, cassava, sweet potato, banana, coffee, cotton)
- Cold Start: 10-15 seconds (model loading)

This architecture represents the complete pipeline from raw agricultural data collection through AI model training to deployed production API serving farmers with intelligent crop recommendations.

