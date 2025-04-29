import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import pandas as pd  # Add this import for handling CSV files
from src.utils.logging_utils import setup_logging

# Configure CUDA settings to avoid device-side assert errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error reporting
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Reduce memory fragmentation

# Choose model type - set to "distilbert" for faster training with good accuracy
MODEL_TYPE = "distilbert"  # Switching to DistilBERT for much faster training

# Fast training configs
BATCH_SIZE = 32  # Larger batch size for faster iteration
EPOCHS = 2  # Reduced epochs for faster completion
USE_SUBSET = True  # Use a subset of data for faster training
SUBSET_SIZE = 40000  # Number of samples to use (20k per class)
SAVE_THRESHOLD = 70.0  # Lower threshold for saving model

# Setup logging
logger = setup_logging("training_logs.log")

class SentimentClassifier(nn.Module):
    """
    Neural network model for sentiment analysis using pre-trained transformer models.
    Can use either BERT or DistilBERT based on configuration.
    """
    def __init__(self, model_type=MODEL_TYPE, hidden_dim=768, output_dim=2):
        super(SentimentClassifier, self).__init__()
        self.model_type = model_type
        
        # Select model based on type
        if model_type == "bert":
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            # Partially freeze BERT (only train the top layers)
            modules = [self.bert.embeddings, *self.bert.encoder.layer[:8]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        else:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            # Freeze DistilBERT parameters to speed up training
            for param in self.bert.parameters():
                param.requires_grad = False
            
        # Classification head - similar structure to what worked well for you previously
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Keep dropout moderate to match your successful model
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings from BERT/DistilBERT
        if self.model_type == "bert":
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output  # BERT provides pooled output directly
        else:
            with torch.no_grad():  # Don't compute gradients for DistilBERT
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                # Use the [CLS] token representation (first token)
                pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classifier
        return self.classifier(pooled_output)

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

def prepare_data(processed_data_path, device, model_type=MODEL_TYPE):
    """Load preprocessed data and prepare for training."""
    print(f"Loading preprocessed data from {processed_data_path}...")
    try:
        data = torch.load(processed_data_path)
        encodings, labels = data['encodings'], data['labels']
        
        # Map the Twitter sentiment labels (0=negative, 4=positive) to binary (0, 1)
        label_mapping = {0: 0, 4: 1}
        labels_cpu = labels.cpu().numpy()
        labels_mapped = torch.tensor([label_mapping.get(int(label), 0) for label in labels_cpu], 
                                  dtype=torch.long)
        
        # Balance the dataset by taking equal samples from each class
        class_0_indices = (labels_mapped == 0).nonzero(as_tuple=True)[0]
        class_1_indices = (labels_mapped == 1).nonzero(as_tuple=True)[0]
        
        # Take a smaller number of samples per class for faster training
        if USE_SUBSET:
            samples_per_class = min(SUBSET_SIZE // 2, len(class_0_indices), len(class_1_indices))
            print(f"FAST MODE: Using {samples_per_class} samples per class ({samples_per_class*2} total)")
        else:
            samples_per_class = min(len(class_0_indices), len(class_1_indices))
            print(f"Using {samples_per_class} samples per class ({samples_per_class*2} total)")
        
        # Randomly sample indices
        np.random.seed(42)
        class_0_sample = np.random.choice(class_0_indices, samples_per_class, replace=False)
        class_1_sample = np.random.choice(class_1_indices, samples_per_class, replace=False)
        
        # Combine indices and get corresponding data
        balanced_indices = torch.cat([torch.tensor(class_0_sample), torch.tensor(class_1_sample)])
        
        input_ids = encodings['input_ids'][balanced_indices].long()  # BERT expects long type
        attention_mask = encodings['attention_mask'][balanced_indices].long()
        balanced_labels = labels_mapped[balanced_indices]
        
        # Split data into training and validation sets
        train_indices, val_indices = train_test_split(
            range(len(balanced_indices)), 
            test_size=0.2,  # 80% train, 20% validation - standard split
            stratify=balanced_labels,
            random_state=42
        )
        
        # Extract train/val data
        train_input_ids = input_ids[train_indices].to(device)
        train_attn_mask = attention_mask[train_indices].to(device)
        train_labels = balanced_labels[train_indices].to(device)
        
        val_input_ids = input_ids[val_indices].to(device)
        val_attn_mask = attention_mask[val_indices].to(device)
        val_labels = balanced_labels[val_indices].to(device)
        
        print(f"Training data shape: {train_input_ids.shape}, Labels: {train_labels.shape}")
        print(f"Validation data shape: {val_input_ids.shape}, Labels: {val_labels.shape}")
        
        # Check class balance
        train_class_counts = torch.bincount(train_labels)
        val_class_counts = torch.bincount(val_labels)
        print(f"Training class distribution: {train_class_counts}")
        print(f"Validation class distribution: {val_class_counts}")
        
        return train_input_ids, train_attn_mask, val_input_ids, val_attn_mask, train_labels, val_labels
    
    except Exception as e:
        print(f"Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def prepare_data_with_new_dataset(raw_data_path, processed_data_path, device, model_type=MODEL_TYPE):
    """Load raw data from CSV, preprocess, and prepare for training."""
    print(f"Loading raw data from {raw_data_path}...")
    try:
        # Load the new dataset
        df = pd.read_csv(raw_data_path)
        
        # Ensure the dataset has the required columns and rename them
        df.columns = ['label', 'id', 'date', 'query', 'user', 'text']
        
        # Map labels to binary (0: Negative, 1: Positive)
        label_mapping = {0: 0, 4: 1}  # Adjust as needed for your dataset
        df['label'] = df['label'].map(label_mapping)
        
        # Tokenize the text data
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if model_type == "bert" else DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        encodings = tokenizer(
            df['text'].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Save the processed data for reuse
        torch.save({'encodings': encodings, 'labels': torch.tensor(df['label'].values)}, processed_data_path)
        print(f"Processed data saved to {processed_data_path}")
        
        # Call the existing prepare_data function to split and balance the data
        return prepare_data(processed_data_path, device, model_type)
    
    except Exception as e:
        print(f"Error preparing data with new dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def train_model(model, train_data, val_data, device, batch_size=BATCH_SIZE, epochs=EPOCHS, model_type=MODEL_TYPE):
    """
    Train the model with batching and progress tracking.
    Optimized for the high accuracy target of 78.09%.
    """
    try:
        # Unpack data
        train_input_ids, train_attn_mask, train_labels = train_data
        val_input_ids, val_attn_mask, val_labels = val_data
        
        # Create data loaders for batching
        train_dataset = TensorDataset(train_input_ids, train_attn_mask, train_labels)
        val_dataset = TensorDataset(val_input_ids, val_attn_mask, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer - similar to what worked well for you before
        if model_type == "bert":
            # Use different learning rates for different parts of the model
            # This is common practice when fine-tuning BERT
            bert_params = list(model.bert.parameters())
            classifier_params = list(model.classifier.parameters())
            
            optimizer = optim.AdamW([
                {'params': bert_params, 'lr': 2e-5},  # Lower learning rate for BERT
                {'params': classifier_params, 'lr': 1e-4}  # Higher learning rate for classifier
            ], weight_decay=0.01)
        else:
            # Simpler optimizer setup for DistilBERT (all classifier parameters)
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        
        # Add learning rate scheduler - cosine schedule with warmup
        # This often produces better results than ReduceLROnPlateau
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warmup
        
        def lr_lambda(current_step):
            # Linear warmup followed by cosine decay
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Track best model
        best_accuracy = 0.0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            
            # Training
            for i, (input_ids, attn_mask, targets) in enumerate(train_loader):
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(input_ids, attn_mask)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                
                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    print(f'Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}: Loss = {running_loss / 10:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}')
                    running_loss = 0.0
                    
                    # Clear GPU cache periodically
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for input_ids, attn_mask, targets in val_loader:
                    outputs = model(input_ids, attn_mask)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    # Store predictions and targets for detailed metrics
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    # Calculate validation loss
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            # Calculate metrics
            epoch_time = time.time() - start_time
            accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(val_loader)
            
            print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s')
            print(f'Validation Accuracy: {accuracy:.2f}%, Loss: {avg_val_loss:.4f}')
            
            # Calculate per-class metrics
            np_preds = np.array(all_predictions)
            np_targets = np.array(all_targets)
            for class_id in range(2):
                class_mask = (np_targets == class_id)
                if np.sum(class_mask) > 0:
                    class_correct = np.sum((np_preds == class_id) & class_mask)
                    class_total = np.sum(class_mask)
                    print(f"  Class {class_id}: {class_correct}/{class_total} = {100*class_correct/class_total:.2f}%")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
        
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"Restored best model with validation accuracy: {best_accuracy:.2f}%")
        
        logger.info("Model training completed successfully.")
        return model, best_accuracy
    except Exception as e:
        logger.error(f"Error during model training: {e}")

from src.utils.sentiment_model_saver import SentimentAnalysisModelWrapper, save_model

def save_model(model, model_type=MODEL_TYPE, output_path='model.pkl'):
    """Save the trained model for application use."""
    print(f"Saving model to {output_path}...")
    
    # Move model to CPU for saving
    model.cpu()
    
    # Create the wrapper model using the utility from sentiment_model_saver.py
    wrapper_model = SentimentAnalysisModelWrapper(model, model_type)
    
    # Save using joblib
    try:
        save_model(wrapper_model, output_path)
        print(f"Model successfully saved to {output_path}")
        
        # Test the saved model
        print("Testing saved model with sample texts...")
        test_model = joblib.load(output_path)
        
        test_texts = [
            "I absolutely love this product, it's amazing!",
            "This is the worst experience I've ever had.",
            "The service was okay, nothing special."
        ]
        
        test_results = test_model.predict(test_texts)
        for text, result in zip(test_texts, test_results):
            sentiment = "Positive" if result == 1 else "Negative"
            print(f"'{text}' → {sentiment}")
        
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False

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

    # Allow user to choose between new dataset and preprocessed data
    use_new_dataset = input("Use new dataset? (yes/no): ").strip().lower() == 'yes'

    if use_new_dataset:
        raw_data_path = os.path.join('data', 'raw', 'sentimentdataset.csv')
        processed_data_path = 'processed_data_with_new_dataset.pt'

        # Prepare data using the new dataset
        train_input_ids, train_attn_mask, val_input_ids, val_attn_mask, train_labels, val_labels = prepare_data_with_new_dataset(
            raw_data_path, processed_data_path, device, model_type=MODEL_TYPE
        )
    else:
        processed_data_path = 'processed_data.pt'

        # Prepare data using preprocessed data
        train_input_ids, train_attn_mask, val_input_ids, val_attn_mask, train_labels, val_labels = prepare_data(
            processed_data_path, device, model_type=MODEL_TYPE
        )

    if train_input_ids is None:
        print("Failed to prepare data. Exiting.")
        sys.exit(1)

    # Initialize model
    try:
        model = SentimentClassifier(model_type=MODEL_TYPE).to(device)
        print(f"Created {MODEL_TYPE.upper()} model and moved to {device}")

        # Train model with optimized fast parameters
        trained_model, accuracy = train_model(
            model, 
            train_data=(train_input_ids, train_attn_mask, train_labels), 
            val_data=(val_input_ids, val_attn_mask, val_labels),
            device=device,
            batch_size=BATCH_SIZE,  # Use the fast batch size constant
            epochs=EPOCHS,  # Use the fast epochs constant
            model_type=MODEL_TYPE
        )

        # Save model with lower threshold for quick results
        if accuracy > SAVE_THRESHOLD:
            save_model(trained_model, model_type=MODEL_TYPE)
        else:
            print(f"Model accuracy ({accuracy:.2f}%) is below target of {SAVE_THRESHOLD}%. Model not saved.")

    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()

    # Final cleanup
    if gpu_available:
        reset_gpu()

if __name__ == "__main__":
    main()