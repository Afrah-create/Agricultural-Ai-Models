# Agricultural AI System - Complete Pipeline Illustration Guide

## Copy-Paste Text for Your Illustration Tool

### OVERALL STRUCTURE: 5 Main Stages (Left to Right)

```
Stage 1: Raw Data Collection → Stage 2: Preprocessing → Stage 3: Knowledge Graph → Stage 4: Model Training → Stage 5: Production API
```

### STAGE 1: RAW DATA SOURCES (Leftmost)

Two parallel data collection sources in separate boxes:

**Box A: Ugandan Dataset**
- Label: "Ugandan Agricultural Dataset"
- Format: CSV file (Ugandan_data.csv)
- Content: Thousands of records
- Fields shown: crop_name, pH, organic_matter, nitrogen, phosphorus, potassium, texture_class, temperature, rainfall, agro_ecological_zone
- Arrow pointing right to Stage 2

**Box B: Literature PDFs**
- Label: "Academic Literature (52 PDFs)"
- Format: PDF files in Literature_reviews/ folder
- Content: Agricultural research papers
- Fields shown: Soil science, crop suitability, climate adaptation
- Arrow pointing right to Stage 2

### STAGE 2: PREPROCESSING (Second from Left)

Large processing box with 7 sub-processes arranged vertically:

**Sub-process 01**: Environment Setup - Install Python packages

**Sub-process 04**: Dataset Preprocessing
- Input: Ugandan_data.csv
- Operations: Clean missing values, normalize crop names, calculate derived features (suitability_score, yield_score, soil_quality_index, climate_suitability, overall_suitability)
- Output: ugandan_data_preprocessed.csv
- Arrow from Box A above

**Sub-process 06**: PDF Text Extraction
- Input: Literature PDFs
- Operations: Extract text using pdfplumber, call Gemini API for structured extraction
- Output: Raw text from PDFs
- Arrow from Box B above

**Sub-process 07**: AI-Powered Triple Extraction
- Input: PDF text
- Operations: Use Gemini API to extract (subject, predicate, object, evidence) triples
- Output: literature_triples.json
- Arrow from Sub-process 06

**Sub-process 05**: Advanced Cleaning
- Operations: Handle duplicates, outliers, data types
- Output: ugandan_data_cleaned.csv
- Arrow from Sub-process 04

### STAGE 3: KNOWLEDGE GRAPH (Center)

Large rectangular box with three sub-processes:

**Sub-process 08**: Dataset to Triples
- Input: ugandan_data_cleaned.csv
- Operations: Convert CSV rows to RDF triples
- Creates: Subject → Predicate → Object relationships
- Examples: "maize → has_nutrient_requirement → nitrogen_level"
- Output: dataset_triples.json (~50,000 triples)
- Arrow from Stage 2

**Sub-process 10**: Triple Integration
- Inputs: dataset_triples.json + literature_triples.json
- Operations: Merge, deduplicate, create unified vocabulary
- Output: unified_knowledge_graph.json (175,318 triples)
- Arrows from Sub-process 08 and 07

**Sub-process 11**: Graph Construction & Visualization
- Input: unified_knowledge_graph.json
- Operations: Build NetworkX graph, calculate metrics, generate schema
- Output: unified_knowledge_graph_schema.json, visualizations
- Arrow from Sub-process 10
- Label: "Knowledge Graph Ready"

### STAGE 4: MODEL TRAINING (Fourth from Left)

Large training box with labeled sub-processes:

**Sub-process 12**: Graph Embedding Training
- Input: Unified knowledge graph (175K triples)
- Process: Train 5 models in parallel
  - Model 1: TransE (Translation-based)
  - Model 2: DistMult (Bilinear)
  - Model 3: ComplEx (Complex embeddings)
  - Model 4: GCN - Graph Convolutional Network (BEST - 87.28% accuracy)
  - Model 5: GraphSAGE (Sampling-based)
- Architecture shown: Input embeddings (2513 entities × 100 dim) → Hidden layer (200 dim) → Output embeddings (100 dim)
- Training: 80% train, 10% validation, 10% test with negative sampling
- Output: best_model.pth, model_metadata.json, all 5 model files
- Label: "BEST MODEL: GCN selected"
- Arrow from Stage 3

**Sub-process 14-25**: Additional Setup (Smaller box below Sub-process 12)
- Set up RAG Pipeline (TF-IDF retrieval)
- Fine-tune DialoGPT LLM (117M parameters)
- Output: quick_fine_tuned_fast/ directory

### STAGE 5: PRODUCTION API (Rightmost)

Five vertical layers shown as a stack with API at top, AI engines in middle:

**Layer 5 (Top): Web Interface**
- HTML form with input fields: pH, organic matter, texture, nitrogen, phosphorus, potassium, temperature, rainfall, available land
- Display area for results
- PDF download button

**Layer 4: API Gateway**
- Three Flask endpoints: Home, /api/recommend, /api/download_pdf
- Handles HTTP requests and returns JSON/PDF

**Layer 3: Core AI Engine (Largest box)**
- Four AI subsystems arranged in 2x2 grid:
  - Top Left: Constraint Engine (Rule-based, evaluates 8 crops)
  - Top Right: RAG Pipeline (TF-IDF retrieval from 175K triples)
  - Bottom Left: GCN Enhancement (Loads best_model.pth, enhances scores)
  - Bottom Right: LLM Generator (Fine-tuned DialoGPT + Gemini API)
- Central hub: AgriculturalAPI orchestrator
- Label: "Multi-AI Ensemble"

**Layer 2: Trained Models**
- Five model files: best_model.pth, transe_model.pth, distmult_model.pth, complex_model.pth, graphsage_model.pth
- Model metadata with entity mappings
- Fine-tuned LLM files

**Layer 1 (Bottom): Infrastructure**
- Docker container
- Gunicorn server
- Railway cloud hosting

### ARROWS AND FLOW DIRECTIONS

**Left to Right (Top)**: 
Raw Data → Preprocessing → Knowledge Graph → Model Training → Production API
(Show as progressively transforming data flow)

**Vertical Flow in Production**:
Layer 5 (User Input) ↓ → Layer 4 (API) ↓ → Layer 3 (AI Processing) → Layer 2 (Models) → Layer 1 (Infrastructure)
Then upward response flow: Layer 1 ↑ → Layer 2 ↑ → Layer 3 ↑ → Layer 4 ↑ → Layer 5 (Display Results)

**Parallel Processing**:
In Layer 3, show all 4 AI subsystems activating simultaneously with arrows from central orchestrator

**Data Loading Arrows**:
From knowledge graph (Stage 3) → RAG Pipeline (Layer 3, Top Right)
From model files (Stage 4) → GCN Enhancement (Layer 3, Bottom Left)

### VISUAL STYLING

**Stage Colors**:
- Stage 1 (Raw Data): Light gray boxes
- Stage 2 (Preprocessing): Blue boxes
- Stage 3 (Knowledge Graph): Green circles/nodes connected in graph structure
- Stage 4 (Model Training): Orange/yellow with gradient
- Stage 5 (Production): Multi-color by layer

**Sizes**:
- Stage 3 Knowledge Graph: Show as network graph with nodes and edges
- Stage 4 Model Training: Show as neural network diagram
- Layer 3 in Stage 5: Largest box showing coordinated parallel processing

**Labels and Numbers**:
- "50K triples" (Stage 3, Sub-process 08)
- "175K unified graph" (Stage 3, Sub-process 10)
- "2513 entities × 100 dim" (Stage 4, architecture)
- "87.28% accuracy" (Stage 4, GCN model)
- "8 crops evaluated" (Stage 5, Constraint Engine)
- "<2 sec response" (Stage 5, performance)

**Data Volume Indicators**:
- Small data: Thin arrows (dataset, model files)
- Large data: Thick arrows (knowledge graph, 175K triples)

### KEY CONNECTIONS TO HIGHLIGHT

1. Ugandan Dataset → Data Preprocessing → Dataset Triples → Unified Graph (blue path)
2. Literature PDFs → AI Extraction → Literature Triples → Unified Graph (green path)
3. Unified Graph → Model Training → Trained Models (thick orange arrow)
4. Trained Models + Unified Graph → Production API (dotted arrows, loaded on request)
5. Production API → Four AI Subsystems → Combined Recommendations (parallel multi-arrow)

This complete architecture shows the full journey from raw agricultural data through sophisticated AI processing to deployed production system serving farmers.

