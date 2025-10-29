# Agricultural AI System - Architecture for Illustration

## Copy-Paste This Text for Your Illustration Tool

### System Overview
Five-layer vertical architecture for an intelligent crop recommendation system. Top to bottom: Web Interface → API Gateway → AI Engine → Data Storage → Infrastructure.

### Layer 1: Web Interface (Top)
- Single responsive web page with input form
- Fields: pH, organic matter, texture, nitrogen, phosphorus, potassium, temperature, rainfall, available land
- Three output elements: recommendation display, suitability scores table, PDF download button

### Layer 2: API Gateway
- Three Flask endpoints in rectangular boxes
- Route 1: Home page (serves HTML)
- Route 2: POST /api/recommend (accepts JSON, validates input, returns recommendations)
- Route 3: POST /api/download_pdf (generates and serves PDF reports)

### Layer 3: Core AI Engine (Largest Middle Layer)
Central hub with four AI subsystems arranged in a 2x2 grid:

**Top Left: Constraint Engine**
- Rule-based validation system
- Contains eight crop constraints (maize, rice, beans, cassava, sweet potato, banana, coffee, cotton)
- Evaluates crops against pH, temperature, rainfall, nutrients
- Outputs suitability scores 0-1

**Top Right: RAG Pipeline**
- Semantic retriever with TF-IDF vectorization
- Connects to 175K triple knowledge graph
- Performs hybrid retrieval to find relevant agricultural evidence
- Returns top 15 relevant triples

**Bottom Left: GCN Enhancement**
- Loads pre-trained Graph Convolutional Network model
- Contains entity embeddings (2513 entities × 100 dimensions)
- Enhances suitability scores using graph structure
- Two-layer GCN architecture (100 → 200 → 100 dimensions)

**Bottom Right: LLM Generation**
- Three-component fallback chain
- Component 1: Fine-tuned DialoGPT model (117M parameters)
- Component 2: Google Gemini API (fallback)
- Component 3: Template-based generator (final fallback)
- Generates AI insights, expert analysis, and implementation plans

**Center: AgriculturalAPI Orchestrator**
- Coordinates all four subsystems
- Receives requests from API layer
- Distributes work to subsystems
- Combines outputs into unified recommendations
- Generates land allocation and evaluation scores

### Layer 4: Data Storage (Four Separate Boxes)

**Box 1: Knowledge Graph**
- 175,318 triples in unified JSON file
- Entity types: crops, soils, zones, practices
- Relationship types: requires_ph, suitable_for, nutrient_requirement
- Connected to RAG Pipeline (Layer 3)

**Box 2: Trained Models**
- Five model files: best_model.pth (GCN), transe_model.pth, distmult_model.pth, complex_model.pth, graphsage_model.pth
- Model metadata JSON with entity mappings
- Connected to GCN Enhancement (Layer 3)

**Box 3: Processed Dataset**
- ugandan_data_cleaned.csv with soil and climate data
- 1000+ records with crop information
- Connected to Constraint Engine (Layer 3)

**Box 4: Fine-Tuned LLM**
- quick_fine_tuned_fast directory
- Model checkpoints and tokenizer files
- Connected to LLM Generation (Layer 3)

### Layer 5: Infrastructure (Bottom)
- Docker container environment
- Gunicorn WSGI server with multiple workers
- Python 3.11 runtime
- Dependencies: PyTorch, scikit-learn, Flask, ReportLab, pandas
- Railway cloud platform hosting

### Data Flow Arrows

**Top-Down Request Flow:**
Layer 1 → Layer 2 (HTTP POST with JSON) → Layer 3 orchestrator → All subsystems activated in parallel

**Upward Response Flow:**
Subsystems → Orchestrator → Layer 2 → Layer 1 (JSON response displayed)

**Diagonal Data Loading:**
Layer 4 repositories → Layer 3 subsystems (dashed lines for lazy loading)

**Horizontal Coordination:**
Subsystems A, B, C, D connected to central orchestrator with bidirectional arrows

### Visual Styling Suggestions
- Layer 3 should be largest box (core processing)
- Use color coding: Blue for presentation, Orange for API, Green for core engine, Yellow for data, Gray for infrastructure
- Show data volumes: Large knowledge graph (175K), small model files
- Indicate parallel processing: Multiple arrows from orchestrator to subsystems simultaneously
- Add labels for key numbers: "2513 entities", "8 crops", "175K triples", "100-dim embeddings"

### Key Metrics to Display
- Knowledge Graph: 175,318 triples
- Model Size: 100-dimensional embeddings
- Entities: 2,513 agricultural entities
- Relations: 15 relationship types
- Supported Crops: 8 major crops
- Response Time: <2 seconds
- Accuracy: GCN model 87.28%

This architecture can be drawn as a multi-layer stack with the AI engine as the central processing hub and four data repositories below it feeding information upward into the processing components.

