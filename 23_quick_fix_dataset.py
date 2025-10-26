"""
QUICK FIX CELL - Run this to resolve the TypeError
This cell fixes the dataset creation issue
"""

# Quick fix for the dataset creation error
if 'text_samples' in locals() and text_samples:
    print("🔧 Applying quick fix for dataset creation...")
    
    # Extract just the text for training
    texts = [sample['text'] for sample in text_samples]
    
    # Split into train/validation
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    print(f"Training samples: {len(train_texts):,}")
    print(f"Validation samples: {len(val_texts):,}")
    
    # Tokenize texts directly
    def tokenize_texts(texts, tokenizer, max_length=512):
        """Tokenize texts and return HuggingFace dataset format"""
        tokenized_data = []
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"  Tokenizing {i+1}/{len(texts)} texts...")
                
            # Tokenize text
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            tokenized_data.append({
                'input_ids': encoding['input_ids'].flatten().tolist(),
                'attention_mask': encoding['attention_mask'].flatten().tolist(),
                'labels': encoding['input_ids'].flatten().tolist()
            })
        
        return tokenized_data
    
    # Tokenize training and validation texts
    print("🔄 Tokenizing training texts...")
    train_tokenized = tokenize_texts(train_texts, tokenizer)
    
    print("🔄 Tokenizing validation texts...")
    val_tokenized = tokenize_texts(val_texts, tokenizer)
    
    # Create HuggingFace datasets
    train_hf_dataset = HFDataset.from_list(train_tokenized)
    val_hf_dataset = HFDataset.from_list(val_tokenized)
    
    print("✅ Datasets prepared for training!")
    print(f"Training dataset size: {len(train_hf_dataset):,}")
    print(f"Validation dataset size: {len(val_hf_dataset):,}")
    
    # Verify dataset structure
    print("\n🔍 Dataset structure verification:")
    sample = train_hf_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Input IDs shape: {len(sample['input_ids'])}")
    print(f"Attention mask shape: {len(sample['attention_mask'])}")
    print(f"Labels shape: {len(sample['labels'])}")
    
    print("\n✅ Quick fix applied successfully!")
    print("You can now proceed with the training cell.")
    
else:
    print("❌ text_samples not found. Please run the dataset creation cell first.")
