"""
Phase 1, Cell 3: Google Drive Mounting and Project Structure Setup
This cell mounts Google Drive and creates the project directory structure
"""

from google.colab import drive
drive.mount('/content/drive')

# Set up project paths
PROJECT_ROOT = '/content/drive/MyDrive/Final'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
KG_DATA_DIR = os.path.join(DATA_DIR, 'knowledge_graph')
LITERATURE_DIR = os.path.join(PROJECT_ROOT, 'Literature_reviews')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'Dataset')

# Create directory structure
directories = [
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    KG_DATA_DIR,
    os.path.join(PROJECT_ROOT, 'src'),
    os.path.join(PROJECT_ROOT, 'src', 'data_extraction'),
    os.path.join(PROJECT_ROOT, 'src', 'knowledge_graph'),
    os.path.join(PROJECT_ROOT, 'src', 'embeddings'),
    os.path.join(PROJECT_ROOT, 'src', 'rag_pipeline'),
    os.path.join(PROJECT_ROOT, 'src', 'llm_training'),
    os.path.join(PROJECT_ROOT, 'src', 'optimization'),
    os.path.join(PROJECT_ROOT, 'src', 'evaluation'),
    os.path.join(PROJECT_ROOT, 'notebooks'),
    os.path.join(PROJECT_ROOT, 'config'),
    os.path.join(PROJECT_ROOT, 'tests'),
    os.path.join(PROJECT_ROOT, 'docs')
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Verify existing directories
print("\nChecking existing directories:")
existing_dirs = [LITERATURE_DIR, DATASET_DIR]
for directory in existing_dirs:
    if os.path.exists(directory):
        files = os.listdir(directory)
        print(f"{directory}: {len(files)} files found")
        if len(files) > 0:
            print(f"   Sample files: {files[:3]}")
    else:
        print(f"{directory}: Not found")

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"Literature directory: {LITERATURE_DIR}")
print(f"Dataset directory: {DATASET_DIR}")
