"""
Phase 4, Cell 1: PDF Triple Extraction from Literature Reviews
This cell extracts structured triples from PDFs in the Literature_reviews folder using Gemini API
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import PDF processing libraries
import PyPDF2
import pdfplumber
from io import BytesIO

# Import Gemini API
import google.generativeai as genai

# Configure Gemini API
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("Gemini API configured successfully!")
except Exception as e:
    print(f"Gemini API configuration failed: {e}")
    print("Trying alternative model...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Using Gemini 1.5 Flash model")
    except Exception as e2:
        print(f"Alternative model also failed: {e2}")

def extract_text_from_pdf(pdf_path, method='pdfplumber'):
    """
    Extract text from PDF using specified method with error handling
    """
    text = ""
    
    try:
        if method == 'pdfplumber':
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_error:
                        print(f"    Warning: Error extracting page {page_num + 1}: {page_error}")
                        continue
        
        elif method == 'PyPDF2':
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_error:
                        print(f"    Warning: Error extracting page {page_num + 1}: {page_error}")
                        continue
        
        return text.strip()
    
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_soil_crop_triples(text, pdf_name):
    """
    Extract soil-crop relationship triples using Gemini API
    """
    
    prompt = f"""
    Extract soil-crop suitability relationships from this agricultural research text.
    
    Focus on extracting structured triples in the following format:
    
    SOIL-CROP SUITABILITY RELATIONSHIPS:
    - Soil properties (pH, EC, CEC, organic matter, texture, nutrients)
    - Crop names and varieties
    - Suitability conditions and thresholds
    - Management recommendations
    
    MANAGEMENT PRACTICES:
    - Soil amendments (lime, gypsum, compost, fertilizers)
    - Application rates and timing
    - Effects on soil properties
    
    CLIMATE REQUIREMENTS:
    - Temperature ranges
    - Rainfall requirements
    - Growing season characteristics
    
    Return a JSON structure with:
    {{
        "soil_crop_relationships": [
            {{
                "subject": "soil_property_or_class",
                "predicate": "relationship_type",
                "object": "crop_name",
                "conditions": "specific_conditions",
                "threshold": "threshold_value",
                "evidence": "supporting_text_excerpt",
                "confidence": 0.0-1.0
            }}
        ],
        "management_practices": [
            {{
                "practice_type": "amendment/fertilization/irrigation",
                "crop": "crop_name",
                "soil_condition": "target_soil_property",
                "application_rate": "rate_and_unit",
                "timing": "application_timing",
                "effect": "expected_outcome",
                "evidence": "supporting_text"
            }}
        ],
        "climate_requirements": [
            {{
                "crop": "crop_name",
                "temperature_range": "min-max_temperature",
                "rainfall_requirement": "rainfall_range",
                "growing_season": "season_characteristics",
                "evidence": "supporting_text"
            }}
        ]
    }}
    
    Text to analyze:
    {text[:4000]}
    """
    
    try:
        response = model.generate_content(prompt)
        
        # Try to parse JSON response
        try:
            import json
            # Clean the response text
            response_text = response.text.strip()
            
            # Remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            structured_data = json.loads(response_text)
            return structured_data
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed for {pdf_name}: {e}")
            # Return raw response with error info
            return {
                "raw_response": response.text,
                "pdf_name": pdf_name,
                "extraction_error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        print(f"Error in Gemini extraction for {pdf_name}: {e}")
        return {
            "error": str(e),
            "pdf_name": pdf_name,
            "timestamp": datetime.now().isoformat()
        }

def process_literature_pdfs():
    """
    Process all PDFs in the Literature_reviews folder
    """
    
    literature_dir = '/content/drive/MyDrive/Final/Literature_reviews'
    
    if not os.path.exists(literature_dir):
        print(f"Literature directory not found: {literature_dir}")
        return []
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(literature_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in Literature_reviews folder")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    extracted_triples = []
    
    # Process first 5 PDFs as a test (you can increase this number)
    max_pdfs = min(5, len(pdf_files))
    print(f"Processing first {max_pdfs} PDFs for testing...")
    
    for i, pdf_file in enumerate(pdf_files[:max_pdfs]):
        print(f"\nProcessing {i+1}/{max_pdfs}: {pdf_file}")
        
        pdf_path = os.path.join(literature_dir, pdf_file)
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            print(f"  Extracted {len(text):,} characters")
            
            # Extract triples using Gemini
            triples = extract_soil_crop_triples(text, pdf_file)
            
            # Add metadata
            triples['pdf_file'] = pdf_file
            triples['text_length'] = len(text)
            triples['processed_at'] = datetime.now().isoformat()
            
            extracted_triples.append(triples)
            
            # Show sample of extracted data
            if 'soil_crop_relationships' in triples and isinstance(triples['soil_crop_relationships'], list):
                print(f"  Extracted {len(triples['soil_crop_relationships'])} soil-crop relationships")
            if 'management_practices' in triples and isinstance(triples['management_practices'], list):
                print(f"  Extracted {len(triples['management_practices'])} management practices")
            if 'climate_requirements' in triples and isinstance(triples['climate_requirements'], list):
                print(f"  Extracted {len(triples['climate_requirements'])} climate requirements")
        else:
            print(f"  No text extracted from {pdf_file}")
    
    print(f"\nProcessed {max_pdfs} PDFs. To process all {len(pdf_files)} PDFs, increase max_pdfs variable.")
    
    return extracted_triples

def analyze_extracted_triples(triples_list):
    """
    Analyze the extracted triples for quality and coverage
    """
    
    print(f"\nTriple Extraction Analysis:")
    print(f"=" * 40)
    
    total_pdfs = len(triples_list)
    successful_extractions = 0
    total_relationships = 0
    total_practices = 0
    total_climate = 0
    
    relationship_types = {}
    crop_coverage = set()
    soil_property_coverage = set()
    
    for triple_data in triples_list:
        if 'soil_crop_relationships' in triple_data and isinstance(triple_data['soil_crop_relationships'], list):
            successful_extractions += 1
            total_relationships += len(triple_data['soil_crop_relationships'])
            
            for rel in triple_data['soil_crop_relationships']:
                if isinstance(rel, dict):
                    # Count relationship types
                    pred = rel.get('predicate', 'unknown')
                    relationship_types[pred] = relationship_types.get(pred, 0) + 1
                    
                    # Track crop coverage
                    if 'object' in rel:
                        crop_coverage.add(rel['object'])
                    if 'subject' in rel:
                        soil_property_coverage.add(rel['subject'])
        
        if 'management_practices' in triple_data and isinstance(triple_data['management_practices'], list):
            total_practices += len(triple_data['management_practices'])
        
        if 'climate_requirements' in triple_data and isinstance(triple_data['climate_requirements'], list):
            total_climate += len(triple_data['climate_requirements'])
    
    print(f"Total PDFs processed: {total_pdfs}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Success rate: {successful_extractions/total_pdfs*100:.1f}%")
    print(f"Total soil-crop relationships: {total_relationships}")
    print(f"Total management practices: {total_practices}")
    print(f"Total climate requirements: {total_climate}")
    
    print(f"\nRelationship Types:")
    for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel_type}: {count}")
    
    print(f"\nCoverage Analysis:")
    print(f"  Unique crops mentioned: {len(crop_coverage)}")
    print(f"  Unique soil properties: {len(soil_property_coverage)}")
    
    if crop_coverage:
        print(f"  Sample crops: {list(crop_coverage)[:10]}")
    if soil_property_coverage:
        print(f"  Sample soil properties: {list(soil_property_coverage)[:10]}")
    
    return {
        'total_pdfs': total_pdfs,
        'successful_extractions': successful_extractions,
        'success_rate': successful_extractions/total_pdfs*100,
        'total_relationships': total_relationships,
        'total_practices': total_practices,
        'total_climate': total_climate,
        'relationship_types': relationship_types,
        'crop_coverage': list(crop_coverage),
        'soil_property_coverage': list(soil_property_coverage)
    }

# Execute PDF triple extraction
print("Starting PDF triple extraction from Literature_reviews folder...")

# Process PDFs
extracted_triples = process_literature_pdfs()

if extracted_triples:
    # Analyze results
    analysis = analyze_extracted_triples(extracted_triples)
    
    # Save extracted triples
    output_path = '/content/drive/MyDrive/Final/data/processed/literature_triples.json'
    with open(output_path, 'w') as f:
        json.dump(extracted_triples, f, indent=2)
    
    print(f"\nExtracted triples saved to: {output_path}")
    
    # Save analysis report
    analysis_path = '/content/drive/MyDrive/Final/data/processed/literature_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis report saved to: {analysis_path}")
    
    # Display sample triples
    print(f"\nSample Extracted Triples:")
    print("-" * 30)
    
    for i, triple_data in enumerate(extracted_triples[:2]):  # Show first 2 PDFs
        print(f"\nPDF {i+1}: {triple_data['pdf_file']}")
        
        if 'soil_crop_relationships' in triple_data and isinstance(triple_data['soil_crop_relationships'], list):
            print("  Soil-Crop Relationships:")
            for rel in triple_data['soil_crop_relationships'][:3]:  # Show first 3
                if isinstance(rel, dict):
                    print(f"    {rel.get('subject', 'N/A')} -> {rel.get('predicate', 'N/A')} -> {rel.get('object', 'N/A')}")
        
        if 'management_practices' in triple_data and isinstance(triple_data['management_practices'], list):
            print("  Management Practices:")
            for practice in triple_data['management_practices'][:2]:  # Show first 2
                if isinstance(practice, dict):
                    print(f"    {practice.get('practice_type', 'N/A')}: {practice.get('crop', 'N/A')}")

else:
    print("No triples extracted. Please check the Literature_reviews folder.")

print(f"\nPDF Triple Extraction Complete!")
print(f"Next steps:")
print(f"  1. Convert cleaned dataset to triples")
print(f"  2. Integrate PDF and dataset triples")
print(f"  3. Build unified knowledge graph")
print(f"  4. Train graph embeddings")
