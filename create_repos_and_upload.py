#!/usr/bin/env python3
"""
Create Hugging Face repositories and upload models
"""

import os
import sys
from huggingface_hub import HfApi

# Get token
token = os.getenv("HF_TOKEN")
if not token:
    print("❌ HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=token)

# Repositories to create
repos = [
    "Awongo/soil-crop-recommendation-model",
    "Awongo/agricultural-llm-finetuned"
]

print("Creating repositories...")
for repo_id in repos:
    repo_name = repo_id.split('/')[-1]
    try:
        print(f"\n📦 Creating repository: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Created: {repo_id}")
    except Exception as e:
        print(f"⚠️  {repo_id}: {e}")

print("\n✅ Repository creation complete!")
print("\nNow run: python upload_to_huggingface.py")

