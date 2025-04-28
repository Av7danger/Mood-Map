import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import joblib

# Configure CUDA settings to avoid device-side assert errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error reporting
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Reduce memory fragmentation

def verify_gpu():
    """Verify GPU is available and configured correctly."""
    print("Checking GPU availability...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU is available: {device_name}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {device_count}")
        
        # Print memory info
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Memory: {memory_allocated:.2f} GB / {memory_total:.2f} GB")
        
        return True
    else:
        print("No GPU detected. Falling back to CPU.")
        return False

def reset_gpu():
    """Reset GPU state to ensure clean operation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("GPU state reset.")

def load_dataset(filepath, sample_size=100000):
    """Load and prepare the dataset."""
    print(f"Loading dataset from {filepath}...")
    
    try:
        # Read CSV
        data = pd.read_csv(filepath, encoding='latin-1', header=None)
        data.columns = ['label', 'id', 'date', 'query', 'user', 'text']
        
        # Take a sample for faster processing
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
        
        print(f"Loaded {len(data)} samples")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def tokenize_data(texts, device, batch_size=1024):
    """Tokenize text data using the DistilBERT tokenizer with GPU acceleration."""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print(f"Using device for tokenization: {device}")
    print(f"Tokenizing {len(texts)} texts in batches of {batch_size}...")
    
    start_time = time.time()
    
    # Process in batches
    num_batches = (len(texts) + batch_size - 1) // batch_size
    all_input_ids = []
    all_attention_mask = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch = texts[start_idx:end_idx]
        
        if i % 10 == 0:
            print(f"Processing batch {i+1}/{num_batches} ({(i+1)/num_batches*100:.1f}%)")
        
        # Tokenize the batch
        batch_encodings = tokenizer(
            batch, 
            truncation=True, 
            padding='max_length', 
            max_length=512, 
            return_tensors='pt'
        )
        
        # Move to device (GPU or CPU)
        batch_encodings = {k: v.to(device) for k, v in batch_encodings.items()}
        
        # Store results (move back to CPU for easier handling)
        all_input_ids.append(batch_encodings['input_ids'].cpu())
        all_attention_mask.append(batch_encodings['attention_mask'].cpu())
        
        # Explicitly free CUDA memory after each batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Combine batches
    encodings = {
        'input_ids': torch.cat(all_input_ids, dim=0),
        'attention_mask': torch.cat(all_attention_mask, dim=0)
    }
    
    tokenization_time = time.time() - start_time
    print(f"Tokenization completed in {tokenization_time:.2f} seconds")
    
    return encodings

def main():
    """Main execution function."""
    # Check if GPU is available
    gpu_available = verify_gpu()
    
    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda') if gpu_available else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Reset GPU state
    if gpu_available:
        reset_gpu()
    
    # Load dataset
    data = load_dataset('backend/training.1600000.processed.noemoticon.csv')
    if data is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Prepare features and labels
    X = data['text'].tolist()
    y = data['label'].tolist()
    
    # Tokenize data
    try:
        encodings = tokenize_data(X, device)
        print(f"Encodings shape: {encodings['input_ids'].shape}")
        
        # Convert labels to tensor
        labels = torch.tensor(y)
        print(f"Labels shape: {labels.shape}")
        
        # Move to device for later processing
        encodings = {k: v.to(device) for k, v in encodings.items()}
        labels = labels.to(device)
        
        # Save processed data for later use
        print("Saving processed data...")
        torch.save({
            'encodings': encodings,
            'labels': labels
        }, 'processed_data.pt')
        
        print("Data processing complete and saved to 'processed_data.pt'")
        
    except Exception as e:
        print(f"Error during processing: {e}")
    
    # Final cleanup
    if gpu_available:
        reset_gpu()

if __name__ == "__main__":
    main()