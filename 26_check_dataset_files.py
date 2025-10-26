# Check Available Dataset Files in Google Drive
# Run this first to see what files are available

import os
import json

def check_available_files():
    """Check what dataset files are available in Google Drive"""
    
    print("🔍 Checking available dataset files...")
    print("=" * 50)
    
    # Check multiple possible locations
    locations_to_check = [
        '/content/drive/MyDrive/Final/data/processed',
        '/content/drive/MyDrive/Final/data',
        '/content/drive/MyDrive/Final',
        '/content/drive/MyDrive',
    ]
    
    found_files = []
    
    for location in locations_to_check:
        if os.path.exists(location):
            print(f"\n📁 Checking: {location}")
            try:
                files = os.listdir(location)
                for file in files:
                    if file.endswith('.json') and ('dataset' in file.lower() or 'text' in file.lower() or 'sample' in file.lower()):
                        full_path = os.path.join(location, file)
                        file_size = os.path.getsize(full_path)
                        print(f"  ✅ {file} ({file_size:,} bytes)")
                        found_files.append(full_path)
            except Exception as e:
                print(f"  ❌ Error reading directory: {e}")
        else:
            print(f"\n❌ Location not found: {location}")
    
    # Check for specific files that might contain text samples
    specific_files = [
        'agricultural_text_dataset.json',
        'text_samples.json',
        'dataset_triples.json',
        'literature_triples.json',
        'unified_knowledge_graph.json'
    ]
    
    print(f"\n🔍 Checking for specific files...")
    for location in locations_to_check:
        if os.path.exists(location):
            for file in specific_files:
                full_path = os.path.join(location, file)
                if os.path.exists(full_path):
                    file_size = os.path.getsize(full_path)
                    print(f"  ✅ {file} at {location} ({file_size:,} bytes)")
                    found_files.append(full_path)
    
    if found_files:
        print(f"\n✅ Found {len(found_files)} relevant files:")
        for file in found_files:
            print(f"  - {file}")
        
        # Try to load and check the content of the first file
        if found_files:
            print(f"\n📊 Checking content of: {found_files[0]}")
            try:
                with open(found_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    print(f"  📝 Contains {len(data)} items")
                    if len(data) > 0:
                        print(f"  🔍 First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                elif isinstance(data, dict):
                    print(f"  📝 Contains {len(data)} keys")
                    print(f"  🔍 Keys: {list(data.keys())}")
                else:
                    print(f"  📝 Data type: {type(data)}")
                    
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
    else:
        print("\n❌ No relevant dataset files found!")
        print("\n💡 You may need to run the dataset creation script first:")
        print("   - Run 19_agricultural_text_dataset.py")
        print("   - Or check if the file was saved with a different name")

if __name__ == "__main__":
    check_available_files()
