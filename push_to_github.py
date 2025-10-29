#!/usr/bin/env python3
"""
Script to push essential files to GitHub Agricultural-Ai-Models repository
Excludes large data files and models (already on Hugging Face)
"""

import subprocess
import os
from pathlib import Path

# Files and directories to include
FILES_TO_INCLUDE = [
    # Core deployment code
    "deployment/app/main.py",
    "deployment/app/__init__.py",
    "deployment/Dockerfile",
    "deployment/requirements.txt",
    "deployment/Procfile",
    "deployment/README.md",
    "deployment/CLOUD_DEPLOYMENT_GUIDE.md",
    "deployment/QUICK_START.md",
    
    # Upload scripts
    "upload_to_huggingface.py",
    "create_repos_and_upload.py",
    
    # Documentation
    "ARCHITECTURE_DESCRIPTION.md",
    "COMPLETE_ARCHITECTURE_FROM_DATA.md",
    "ILLUSTRATION_FULL_PIPELINE.md",
    "ILLUSTRATION_READY_ARCHITECTURE.md",
    "HUGGINGFACE_UPLOAD_GUIDE.md",
    "UPLOAD_COMPLETE.md",
    "INTERFACE_IMPROVEMENTS.md",
    "PIE_CHART_IMPLEMENTATION.md",
    
    # Git config
    ".gitignore",
    
    # README for the repo
    "README.md"
]

def run_command(cmd):
    """Run a git command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def main():
    print("="*70)
    print("🚀 Pushing to GitHub: Agricultural-Ai-Models")
    print("="*70)
    
    # Check if we're in a git repo
    if not run_command("git status"):
        print("❌ Not a git repository or git not installed")
        return
    
    # Create/update README.md for the models repo
    readme_content = """# Agricultural AI Models

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
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✓ Created README.md\n")
    
    # Add files
    print("📦 Staging files...")
    for file_path in FILES_TO_INCLUDE:
        if os.path.exists(file_path):
            run_command(f'git add "{file_path}"')
            print(f"  ✓ Added: {file_path}")
        else:
            print(f"  ⚠ Skipped (not found): {file_path}")
    
    # Add deployment directory structure (but exclude large files via .gitignore)
    print("\n📦 Adding deployment directory (excluding large files)...")
    run_command("git add deployment/")
    
    print("\n✓ Files staged")
    
    # Check status
    result = subprocess.run("git status --short", shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("\n📋 Files to be committed:")
        print(result.stdout)
        
        # Ask for commit message
        commit_msg = "Add deployment code and documentation for Agricultural AI Models"
        
        # Commit
        print(f"\n💾 Committing with message: {commit_msg}")
        if run_command(f'git commit -m "{commit_msg}"'):
            print("✓ Committed successfully")
            
            # Push to models remote
            print(f"\n📤 Pushing to models remote (GitHub)...")
            if run_command("git push models main"):
                print("✅ Successfully pushed to GitHub!")
                print(f"\n🔗 Repository: https://github.com/Afrah-create/Agricultural-Ai-Models")
            else:
                print("❌ Push failed. You may need to authenticate with GitHub.")
        else:
            print("❌ Commit failed")
    else:
        print("\n⚠ No changes to commit (all files may already be committed)")

if __name__ == "__main__":
    main()

