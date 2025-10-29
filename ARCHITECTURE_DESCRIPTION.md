# Agricultural AI System - Architecture Description
## Complete System Architecture in Words for Illustration

### OVERALL SYSTEM STRUCTURE

The Agricultural Recommendation System is a multi-layered AI platform that combines several AI paradigms to provide intelligent crop recommendations for Ugandan farmers. The entire system can be visualized as five horizontal layers connected vertically.

### LAYER 1: PRESENTATION LAYER (Top Layer)
This is the user interface layer at the top of the system. It contains a single web interface built with HTML, CSS, and JavaScript. This is a responsive web page with a form where farmers input their soil properties and climate data. The interface features input fields for pH levels, organic matter percentage, soil texture class, nutrient levels (nitrogen, phosphorus, potassium), temperature, and rainfall. There are also optional fields for farming conditions like available land area, organic farming preferences, and irrigation availability. The presentation layer has three main interaction points: a form submission button that sends data to the API, a results display area that shows the recommendations in a structured format, and a PDF download button that generates downloadable reports. The interface communicates with Layer 2 through HTTP requests.

### LAYER 2: API GATEWAY LAYER (Second Layer from Top)
This layer contains three Flask API endpoints that act as entry points to the system. The first endpoint is the home route which serves the web interface HTML page. The second endpoint is the POST /api/recommend endpoint which accepts JSON data containing soil properties and climate conditions, validates the input data, calls the core engine in Layer 3, and returns structured JSON responses with crop recommendations. The third endpoint is POST /api/download_pdf which accepts the same input data but generates and returns a PDF report instead of JSON. This layer handles request validation, ensures all required fields are present, catches errors gracefully, and formats responses consistently. It sits directly below the presentation layer and above the core processing layer.

### LAYER 3: CORE AI ENGINE LAYER (Middle Layer)
This is the heart of the system, represented as a large rectangular box containing multiple interconnected AI components. This layer consists of four main subsystems operating in parallel and coordinated through a central orchestrator called the AgriculturalAPI class.

**Subsystem A: Constraint-Based Reasoning Engine**
This is positioned on the left side of the core engine. It contains the Agricultural Constraint Engine which stores rule-based constraints for eight crops: maize, rice, beans, cassava, sweet potato, banana, coffee, and cotton. Each crop has constraints like pH range (e.g., maize needs 5.5-7.5), optimal temperature range, rainfall requirements, soil texture preferences, and NPK nutrient thresholds. The engine evaluates each crop against the provided soil and climate data, calculates suitability scores from 0 to 1, generates specific recommendations for soil improvement, and identifies constraint violations. This subsystem operates as a first-pass filter that rapidly evaluates crop compatibility.

**Subsystem B: Graph Neural Network Enhancement**
This component sits in the center-left of Layer 3. It includes the Agricultural Model Loader which loads a pre-trained Graph Convolutional Network (GCN) model from disk. The GCN model has three key components: entity embeddings containing 2513 agricultural entities (crops, soils, zones) represented as 100-dimensional vectors, relation embeddings for 15 different relationship types, and two GCN layers that transform embeddings through graph convolution operations. When a crop passes through Subsystem A, this enhancement system looks up the crop in its entity mapping, retrieves its learned embedding vector, uses the GCN layers to create an enriched representation that captures graph-structured knowledge, and then adjusts the initial suitability score upward based on the model's confidence. This is visualized as bidirectional arrows connecting Subsystem A and B, showing that B enhances the results from A.

**Subsystem C: Retrieval-Augmented Generation (RAG)**
Located in the center-right of Layer 3. It consists of the Semantic Retriever component which implements a hybrid retrieval system. This subsystem connects to a knowledge graph containing 175,318 triples in the format subject-predicate-object (e.g., "maize requires_ph 6.0" or "loam suitable_for maize"). The system uses TF-IDF vectorization to create text representations of all triples. When generating recommendations, it converts the user's query (soil properties plus crop names) into a TF-IDF vector, computes cosine similarity scores with all triple texts, retrieves the top 15 most relevant triples as evidence, and passes this context to Subsystem D for augmented generation. This subsystem is connected to both the knowledge graph data source below it (in Layer 4) and Subsystem D above it.

**Subsystem D: Language Model Generation**
Positioned on the right side of Layer 3. This subsystem has two LLM components operating in a fallback hierarchy. The primary component is a Fine-Tuned LLM (DialoGPT-small, 117M parameters) that was trained specifically on agricultural domain data, including knowledge graph triples, literature reviews, and agricultural facts. When active, it receives contextual information from Subsystem C's RAG retrieval, generates natural language AI insights, expert analysis paragraphs, and implementation recommendations using learned agricultural terminology. The fallback component is the Google Gemini API integration which provides similar capabilities when the fine-tuned model is unavailable. There's also a third-level fallback called the "Improved Template-Based Generator" which creates structured recommendations based on rule-based templates when both LLM components are unavailable. The arrows flow upward from this subsystem to indicate it generates the final recommendation text.

**Central Orchestrator: AgriculturalAPI**
At the center of Layer 3 sits the AgriculturalAPI class, represented as a central hub with connections to all four subsystems. This orchestrator coordinates the entire recommendation process. When a request arrives from Layer 2, it first calls Subsystem A to evaluate all crops and get initial suitability scores. Then it calls Subsystem B to enhance the scores with graph embeddings. It then passes the top crops to Subsystem D for natural language generation. Throughout this process, Subsystem C provides additional evidence through RAG retrieval when needed. The orchestrator also generates additional outputs like land allocation plans (if available land is provided), multi-dimensional evaluation scores (economic, environmental, social, and risk dimensions), and structured recommendation sections. Finally, it combines everything into a unified response that flows back up to Layer 2.

### LAYER 4: DATA STORAGE LAYER (Second from Bottom)
This layer contains persistent data storage organized into four distinct data repositories, each represented as separate boxes connected to Layer 3 above through data loading arrows.

**Data Repository 1: Knowledge Graph Storage**
This is the largest data repository, storing the unified knowledge graph as a JSON file containing 175,318 triples. The knowledge graph represents agricultural entities (crops, soils, zones, management practices) and their relationships. It's connected to Subsystem C (RAG) in Layer 3 via a data loading pipeline.

**Data Repository 2: Model Files**
This repository stores trained machine learning models on disk. It contains five model files: best_model.pth (GCN model), model_metadata.json (entity and relation mappings), transe_model.pth, distmult_model.pth, complex_model.pth, and graphsage_model.pth. It also stores the model metadata containing entity_to_id mappings, id_to_entity mappings, and similar mappings for relations. This repository is connected to Subsystem B (GNN Enhancement) in Layer 3.

**Data Repository 3: Processed Dataset**
This contains cleaned and preprocessed CSV files. The ugandan_data_cleaned.csv file has columns for soil properties, climate conditions, and crop records. This dataset provides training data for the models and reference data for the constraint engine. It's connected to both Subsystem A and Subsystem B in Layer 3.

**Data Repository 4: Fine-Tuned LLM Model**
This repository contains the fine-tuned DialoGPT model files stored in the quick_fine_tuned_fast directory. It includes model checkpoints (checkpoint-1 and checkpoint-500), the main model files (model.safetensors), and tokenizer files (tokenizer.json, vocab.json, config.json). This repository is connected to Subsystem D in Layer 3.

### LAYER 5: INFRASTRUCTURE LAYER (Bottom Layer)
This is the deployment and runtime infrastructure layer. It contains the system's runtime environment, represented as a container or server box. This layer includes the Docker container that encapsulates the entire application, the Gunicorn web server that runs the Flask application with multiple worker processes, the Python 3.11 runtime environment, and all the dependencies like PyTorch for deep learning, scikit-learn for machine learning utilities, Flask for the web framework, ReportLab for PDF generation, and pandas for data manipulation. This infrastructure layer runs on Railway cloud platform (or any cloud provider) and connects to all the data repositories in Layer 4 to load data into memory during startup.

### DATA FLOW AND INTERACTION PATTERNS

**Request Flow (User to System)**:
The flow starts when a farmer fills out the web form in Layer 1 with their soil data. This data flows down through Layer 2 where it's validated by the API endpoints, then into Layer 3 where the central orchestrator receives it. The orchestrator simultaneously activates Subsystems A, B, C, and D. Subsystem A pulls reference data from Repository 3 in Layer 4, evaluates all crops rapidly, and produces suitability scores. Subsystem B then loads the GCN model weights from Repository 2 in Layer 4, enhances the scores from Subsystem A, and produces refined scores. Subsystem C loads the knowledge graph from Repository 1 in Layer 4, performs semantic retrieval, and passes evidence to Subsystem D. Subsystem D loads its fine-tuned model from Repository 4 in Layer 4, generates natural language insights, and combines everything into final recommendations. The flow then reverses, moving up through Layer 3 (central orchestrator packages everything), then Layer 2 (API formats the JSON response), and finally Layer 1 (the interface displays the results to the farmer).

**PDF Generation Flow**:
For PDF download requests, the flow is similar but after reaching Subsystem D, instead of returning JSON, the system calls a specialized PDF generator component (also in Layer 3) which uses ReportLab to create a structured PDF document with sections for title page, soil analysis, crop recommendations, implementation plan, and data sources.

**Lazy Loading Pattern**:
The system uses lazy loading to optimize memory usage. During initial startup in Layer 5, the infrastructure loads but only loads essential components. Heavy models and large knowledge graphs are not loaded into memory until the first actual request arrives. This is represented as dashed arrows from Layer 4 to Layer 3, indicating conditional loading. Subsystem B loads the GCN model only when first needed. This prevents memory exhaustion at startup.

### KEY ARCHITECTURAL PATTERNS

**Multi-AI Ensemble Pattern**: The system combines four different AI paradigms operating in parallel. Subsystem A provides rule-based reasoning with hardcoded constraints, Subsystem B provides deep learning through graph neural networks, Subsystem C provides information retrieval through RAG, and Subsystem D provides natural language generation through fine-tuned and pre-trained LLMs. These diverse approaches work together to produce more robust recommendations than any single approach could achieve alone.

**Fallback Chain Pattern**: In Subsystem D (Language Models), there's a three-level fallback chain. First it tries the fine-tuned DialoGPT model, if that fails it tries Gemini API, if that also fails it uses template-based generation. This ensures the system always provides recommendations even when some AI services are unavailable.

**Data Pipeline Pattern**: Data flows from multiple repositories in Layer 4 up to the processing components in Layer 3. The flow is unidirectional during data loading (layer 4 to layer 3) but bidirectional during inference (layer 3 requests specific data from layer 4 as needed).

**Stateless API Pattern**: Layer 2's API endpoints are stateless - each request is independent and contains all necessary information. The session state is maintained in Layer 3 within the AgriculturalAPI instance which keeps models and data in memory.

### CONNECTION TYPES AND DATA VOLUME

**Direct Connections**: Solid arrows between Layer 1 and Layer 2 represent HTTP connections. Between Layer 2 and Layer 3, they represent function calls. Between Layer 3 and Layer 4, they represent file I/O operations. Between Layer 4 and Layer 5, they represent filesystem access.

**Batch Loading Arrows**: Large arrows from Layer 4 to Layer 3 during initial data loading show substantial data movement (175K triples from knowledge graph, millions of parameters from model files).

**Real-time Query Arrows**: Smaller, thinner arrows during request processing show lightweight queries and responses flowing through the system.

### SCALABILITY MECHANISMS

The system can be scaled horizontally by running multiple instances of the entire stack in parallel, with a load balancer distributing requests across instances. The stateless design in Layer 2 makes this feasible. Vertical scaling involves increasing the container resources in Layer 5 to handle larger workloads and load models faster.

### SECURITY BOUNDARIES

The system has implicit security boundaries. Layer 1 (presentation) is publicly accessible over HTTP. Layer 2 (API) has input validation and sanitization. Layer 3 (core engine) is isolated within the container with no direct external access. Layer 4 (data) contains sensitive models and data files protected by the container's filesystem permissions. Layer 5 (infrastructure) uses environment variables for sensitive configuration like the Gemini API key.

### DEPENDENCIES AND TECHNOLOGY STACK

**Frontend Technologies**: HTML5, CSS3, JavaScript (no frameworks).

**Backend Framework**: Flask 3.1.2 (Python web framework).

**Deep Learning**: PyTorch 2.0.1 (for GCN models), Transformers library (for LLMs).

**Machine Learning Utilities**: scikit-learn 1.3.0 (for TF-IDF), NumPy 1.24.3 (for numerical operations).

**AI Services**: Google Gemini API (external LLM service), Fine-tuned DialoGPT (on-device LLM).

**Data Processing**: Pandas 2.0.3 (for CSV handling), NetworkX (for graph operations).

**Reporting**: ReportLab 4.0.4 (for PDF generation).

**Deployment**: Docker (containerization), Gunicorn 21.2.0 (WSGI server), Railway platform (cloud hosting).

**Python Version**: 3.11.

This complete architecture can be visualized as five horizontal layers stacked vertically, with the presentation layer at the top, API gateway below that, core AI engine in the middle, data storage below that, and infrastructure at the bottom. The core engine layer has four subsystems arranged in a 2x2 grid, connected to the central orchestrator. Data flows vertically during both initial loading and request processing, with horizontal coordination between subsystems within Layer 3.

