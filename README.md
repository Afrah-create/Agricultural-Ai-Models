# Agricultural AI Models

This repository contains the deployment code and documentation for the Agricultural AI recommendation system.

## Models on Hugging Face

The trained models are hosted on Hugging Face Hub:

- **Graph Embedding Models**: [Awongo/soil-crop-recommendation-model](https://huggingface.co/Awongo/soil-crop-recommendation-model)
- **Fine-Tuned LLM**: [Awongo/agricultural-llm-finetuned](https://huggingface.co/Awongo/agricultural-llm-finetuned)

## Architecture

See [COMPLETE_ARCHITECTURE_FROM_DATA.md](COMPLETE_ARCHITECTURE_FROM_DATA.md) for full system architecture.

## Deployment

See [deployment/README.md](deployment/README.md) for deployment instructions.

## Features

- Multi-AI ensemble (GCN, Constraint Engine, RAG, LLM)
- Real-time crop recommendations
- Interactive web interface
- PDF report generation
- Knowledge graph with 175K+ triples

## Quick Start

1. Install dependencies: `pip install -r deployment/requirements.txt`
2. Deploy to Railway or run locally: `cd deployment && python app/main.py`

See [deployment/QUICK_START.md](deployment/QUICK_START.md) for details.

## Model Upload

To upload models to Hugging Face, see [HUGGINGFACE_UPLOAD_GUIDE.md](HUGGINGFACE_UPLOAD_GUIDE.md)
