"""
Phase 4, Cell 2: Complete PDF Triple Extraction (All PDFs)
This cell processes ALL PDFs in the Literature_reviews folder for comprehensive triple extraction
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

def process_all_literature_pdfs():
    """
    Process ALL PDFs in the Literature_reviews folder
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
    
    # Process ALL PDFs
    print(f"\nProcessing ALL {len(pdf_files)} PDFs...")
    
    for i, pdf_file in enumerate(pdf_files):
        print(f"\nProcessing {i+1}/{len(pdf_files)}: {pdf_file}")
        
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
    
    print(f"\nCompleted processing all {len(pdf_files)} PDFs!")
    
    return extracted_triples

def analyze_complete_extraction(triples_list):
    """
    Analyze the complete extraction results
    """
    
    print(f"\nComplete Triple Extraction Analysis:")
    print(f"=" * 50)
    
    total_pdfs = len(triples_list)
    successful_extractions = 0
    total_relationships = 0
    total_practices = 0
    total_climate = 0
    
    relationship_types = {}
    crop_coverage = set()
    soil_property_coverage = set()
    management_types = set()
    
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
            
            for practice in triple_data['management_practices']:
                if isinstance(practice, dict) and 'practice_type' in practice:
                    management_types.add(practice['practice_type'])
        
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
    
    print(f"\nManagement Practice Types:")
    for mgmt_type in sorted(management_types):
        print(f"  {mgmt_type}")
    
    print(f"\nCoverage Analysis:")
    print(f"  Unique crops mentioned: {len(crop_coverage)}")
    print(f"  Unique soil properties: {len(soil_property_coverage)}")
    print(f"  Unique management practices: {len(management_types)}")
    
    if crop_coverage:
        print(f"  Sample crops: {list(crop_coverage)[:15]}")
    if soil_property_coverage:
        print(f"  Sample soil properties: {list(soil_property_coverage)[:15]}")
    
    return {
        'total_pdfs': total_pdfs,
        'successful_extractions': successful_extractions,
        'success_rate': successful_extractions/total_pdfs*100,
        'total_relationships': total_relationships,
        'total_practices': total_practices,
        'total_climate': total_climate,
        'relationship_types': relationship_types,
        'management_types': list(management_types),
        'crop_coverage': list(crop_coverage),
        'soil_property_coverage': list(soil_property_coverage)
    }

# Execute complete PDF triple extraction
print("Starting COMPLETE PDF triple extraction from Literature_reviews folder...")
print("This will process ALL 52 PDFs - this may take some time!")

# Process all PDFs
extracted_triples = process_all_literature_pdfs()

if extracted_triples:
    # Analyze results
    analysis = analyze_complete_extraction(extracted_triples)
    
    # Save extracted triples
    output_path = '/content/drive/MyDrive/Final/data/processed/complete_literature_triples.json'
    with open(output_path, 'w') as f:
        json.dump(extracted_triples, f, indent=2)
    
    print(f"\nComplete extracted triples saved to: {output_path}")
    
    # Save analysis report
    analysis_path = '/content/drive/MyDrive/Final/data/processed/complete_literature_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Complete analysis report saved to: {analysis_path}")
    
    # Display comprehensive summary
    print(f"\nComprehensive Summary:")
    print("-" * 30)
    print(f"Total PDFs processed: {analysis['total_pdfs']}")
    print(f"Success rate: {analysis['success_rate']:.1f}%")
    print(f"Total soil-crop relationships: {analysis['total_relationships']}")
    print(f"Total management practices: {analysis['total_practices']}")
    print(f"Total climate requirements: {analysis['total_climate']}")
    print(f"Unique crops covered: {len(analysis['crop_coverage'])}")
    print(f"Unique soil properties: {len(analysis['soil_property_coverage'])}")
    print(f"Unique management practices: {len(analysis['management_types'])}")
    
    # Show top relationship types
    print(f"\nTop Relationship Types:")
    for rel_type, count in sorted(analysis['relationship_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {rel_type}: {count}")
    
    # Show sample crops
    print(f"\nSample Crops Extracted:")
    for crop in sorted(analysis['crop_coverage'])[:20]:
        print(f"  {crop}")

else:
    print("No triples extracted. Please check the Literature_reviews folder.")

print(f"\nComplete PDF Triple Extraction Finished!")
print(f"Next steps:")
print(f"  1. Convert cleaned dataset to triples")
print(f"  2. Integrate PDF and dataset triples")
print(f"  3. Build unified knowledge graph")
print(f"  4. Train graph embeddings")
