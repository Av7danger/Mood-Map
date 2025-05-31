import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
import joblib
import numpy as np
import torchmetrics
from transformers import (
    BertTokenizer, BertModel, BertConfig,
    DistilBertTokenizer, DistilBertModel, DistilBertConfig,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
import pandas as pd
from src.utils.logging_utils import setup_logging
import gc  # For explicit garbage collection
import math
from tqdm import tqdm  # For progress bars

# Import additional libraries for advanced optimization
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast, GradScaler
import math
from torch.nn.utils import clip_grad_norm_
import random
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import logging
# Replace scikit-learn with torchmetrics
import torchmetrics
# AdamW is now recommended to be imported from torch.optim instead of transformers
from torch.optim import AdamW

# Constants for advanced optimization techniques
DROPOUT_RATE = 0.1  # Dropout rate for model regularization
EMA_DECAY = 0.999  # Exponential Moving Average decay rate
WARMUP_RATIO = 0.1  # Percentage of training steps for warmup
LR_SCHEDULER = "cosine_warmup"  # Type of learning rate scheduler
LAYERWISE_LR_DECAY = 0.9  # Decay factor for layerwise learning rates
MAX_GRAD_NORM = 1.0  # Maximum gradient norm for clipping
ADAM_EPSILON = 1e-8  # Epsilon for Adam optimizer stability
WEIGHT_DECAY = 0.01  # Weight decay for regularization
MODEL_TYPE = "distilbert"  # Default model type
BATCH_SIZE = 32  # Default batch size
EPOCHS = 5  # Default number of epochs
EARLY_STOPPING_PATIENCE = 3  # Default patience for early stopping
CHECKPOINT_DIR = "checkpoints"  # Directory to save checkpoints
GRADIENT_ACCUMULATION_STEPS = 4  # Steps for gradient accumulation
SAVE_THRESHOLD = 80.0  # Minimum accuracy to save the model
USE_SUBSET = False  # Whether to use a subset of data for faster training
SUBSET_SIZE = 10000  # Size of the subset if USE_SUBSET is True
LARGE_TWITTER_DATASET = "data/raw/twitter_1.6m.csv"  # Path to the large Twitter dataset
CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N steps
MIXED_PRECISION = True  # Use mixed precision training
AUGMENT_MISSING_CLASSES = True  # Add synthetic examples for missing classes

# Create the checkpoints directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

from collections import defaultdict
import time
from torch.cuda.amp import autocast, GradScaler
# Replace scikit-learn metrics with torchmetrics
from torchmetrics.classification import F1Score, ConfusionMatrix

# Focal Loss implementation for handling class imbalance
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    Focuses more on hard examples
    """
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = ((1 - pt) ** self.gamma) * BCE_loss
        return focal_loss.mean()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Helper function for reproducibility
def set_seed(seed):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Exponential Moving Average implementation
class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA shadow weights with current model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

USE_LOOKAHEAD = True  # Lookahead optimizer wrapper
USE_SWA = True  # Stochastic Weight Averaging for better generalization
SWA_START = 0.75  # Start SWA after 75% of training
SWA_LR = 1e-5  # SWA learning rate
SWA_ANNEAL_EPOCHS = 3  # Number of epochs to anneal the learning rate
USE_RADAM = True  # Use RAdam optimizer for better convergence
USE_DYNAMIC_PADDING = True  # Use dynamic padding for better efficiency
FREEZE_EMBEDDINGS = True  # Freeze embeddings for faster training
USE_BERT_SPECIFIC_ADAM = True  # Use BertAdam optimizer for better convergence
USE_FOCAL_LOSS = True  # Use focal loss for better handling of class imbalance
FOCAL_LOSS_GAMMA = 2.0  # Focal loss gamma parameter
USE_DYNAMIC_QUANTIZATION = False  # Apply dynamic quantization to model (for deployment only)

# Setup logging
logger = setup_logging("logs/training_logs.log")

class SentimentClassifier(nn.Module):
    """
    Enhanced neural network model for sentiment analysis using pre-trained transformer models.
    Features improved architecture with residual connections and layer normalization.
    Now configured for 3-class sentiment classification (0: Negative, 1: Neutral, 2: Positive)
    """
    def __init__(self, model_type=MODEL_TYPE, hidden_dim=768, output_dim=3):
        super(SentimentClassifier, self).__init__()
        self.model_type = model_type
        
        # Select model based on type with optimized configurations
        if model_type == "bert":
            # Use config to customize BERT for our task
            config = BertConfig.from_pretrained('bert-base-uncased')
            config.hidden_dropout_prob = DROPOUT_RATE
            config.attention_probs_dropout_prob = DROPOUT_RATE
            self.bert = BertModel.from_pretrained('bert-base-uncased', config=config)
            
            # Partial freezing strategy - only train the top layers for efficiency
            modules = [self.bert.embeddings]
            for i in range(8):  # Freeze first 8 layers (out of 12)
                modules.append(self.bert.encoder.layer[i])
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        else:
            # Use config to customize DistilBERT for our task
            config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
            config.dropout = DROPOUT_RATE
            config.attention_dropout = DROPOUT_RATE
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
            
            # Selective layer freezing for DistilBERT
            # Freeze all layers except the top ones
            for param in self.bert.parameters():
                param.requires_grad = False
            
            # Only fine-tune the last 2 transformer layers (of 6 total)
            for i in range(4, 6):
                for param in self.bert.transformer.layer[i].parameters():
                    param.requires_grad = True
        
        # Enhanced classifier with residual connections and layer norm
        self.pre_classifier = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        
        # Middle layer with residual connection
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),  # GELU activation (used in BERT) instead of ReLU
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        
        # Output classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights similar to BERT initialization"""
        modules = [self.pre_classifier, self.middle[0], self.middle[3], self.classifier]
        for module in modules:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings from BERT/DistilBERT
        if self.model_type == "bert":
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
        else:
            # For DistilBERT, process with selected unfrozen layers
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token output
            pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply enhanced classifier with residual connections
        residual = pooled_output
        
        # First block with residual
        x = self.pre_classifier(pooled_output)
        x = self.layer_norm1(x + residual)  # Residual connection
        x = self.dropout1(x)
        
        # Second block with residual
        residual = x
        x = self.middle(x)
        x = self.layer_norm2(x + residual)  # Residual connection
        x = self.dropout2(x)
        
        # Final classification
        return self.classifier(x)

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
        print("No GPU detected. Training on CPU will be significantly slower.")
        user_input = input("Would you like to continue with CPU training? (y/n): ").strip().lower()
        return user_input == 'y'

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
        # Add BatchEncoding to safe globals to fix the serialization issue
        import torch.serialization
        from transformers.tokenization_utils_base import BatchEncoding
        torch.serialization.add_safe_globals([BatchEncoding])
        
        # Now load the data with weights_only=False to allow loading BatchEncoding objects
        data = torch.load(processed_data_path, weights_only=False)
        
        encodings, labels = data['encodings'], data['labels']
        
        # Check class distribution
        class_counts = torch.bincount(labels)
        print(f"Raw class distribution: {class_counts}")
        
        if len(class_counts) < 2 or 0 in class_counts:
            print("Warning: Missing at least one class. Will use all available data without balancing.")
            # Use all data without balancing
            input_ids = encodings['input_ids'].long()  # BERT expects long type
            attention_mask = encodings['attention_mask'].long()
            
            # Split data into training and validation sets without stratification
            dataset_size = len(labels)
            val_size = int(dataset_size * 0.2)  # 20% for validation
            train_size = dataset_size - val_size
            train_dataset, val_dataset = random_split(
                range(dataset_size), 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            train_indices = list(train_dataset)
            val_indices = list(val_dataset)
        else:
            # Balance the dataset by taking equal samples from each class
            class_0_indices = (labels == 0).nonzero(as_tuple=True)[0]
            class_1_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
            # Take a smaller number of samples per class for faster training
            if USE_SUBSET:
                samples_per_class = min(SUBSET_SIZE // 2, len(class_0_indices), len(class_1_indices))
                print(f"FAST MODE: Using {samples_per_class} samples per class ({samples_per_class*2} total)")
            else:
                samples_per_class = min(len(class_0_indices), len(class_1_indices))
                print(f"Using {samples_per_class} samples per class ({samples_per_class*2} total)")
            
            # Randomly sample indices
            torch.manual_seed(42)
            class_0_sample = class_0_indices[torch.randperm(len(class_0_indices))[:samples_per_class]]
            class_1_sample = class_1_indices[torch.randperm(len(class_1_indices))[:samples_per_class]]
            
            # Combine indices and get corresponding data
            balanced_indices = torch.cat([class_0_sample, class_1_sample])
            
            input_ids = encodings['input_ids'][balanced_indices].long()  # BERT expects long type
            attention_mask = encodings['attention_mask'][balanced_indices].long()
            balanced_labels = labels[balanced_indices]
            
            # Create a pytorch Dataset and use random_split with stratification
            dataset_size = len(balanced_indices)
            val_size = int(dataset_size * 0.2)  # 20% for validation
            train_size = dataset_size - val_size
            
            # Split the indices
            indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42))
            
            # Ensure stratified split by separating classes first, then splitting each
            class_0_indices_local = (balanced_labels == 0).nonzero(as_tuple=True)[0]
            class_1_indices_local = (balanced_labels == 1).nonzero(as_tuple=True)[0]
            
            # Calculate split sizes for each class
            class_0_val_size = int(len(class_0_indices_local) * 0.2)
            class_1_val_size = int(len(class_1_indices_local) * 0.2)
            
            # Split indices for each class
            val_indices_class_0 = class_0_indices_local[:class_0_val_size]
            train_indices_class_0 = class_0_indices_local[class_0_val_size:]
            
            val_indices_class_1 = class_1_indices_local[:class_1_val_size]
            train_indices_class_1 = class_1_indices_local[class_1_val_size:]
            
            # Combine train and val indices
            train_indices = torch.cat([train_indices_class_0, train_indices_class_1])
            val_indices = torch.cat([val_indices_class_0, val_indices_class_1])
            
            # Use the balanced labels
            labels = balanced_labels
        
        # Extract train/val data
        train_input_ids = encodings['input_ids'][train_indices].to(device)
        train_attn_mask = encodings['attention_mask'][train_indices].to(device)
        train_labels = labels[train_indices].to(device)
        
        val_input_ids = encodings['input_ids'][val_indices].to(device)
        val_attn_mask = encodings['attention_mask'][val_indices].to(device)
        val_labels = labels[val_indices].to(device)
        
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
        
        # Check the actual columns in the dataset
        print(f"Dataset columns: {df.columns.tolist()}")
        
        # Extract text and sentiment from the appropriate columns
        # For the new dataset format with 15 columns
        if 'Text' in df.columns and 'Sentiment' in df.columns:
            print("Using 'Text' and 'Sentiment' columns from the dataset")
            
            # Drop rows with NaN values in Text or Sentiment
            initial_count = len(df)
            df = df.dropna(subset=['Text', 'Sentiment'])
            print(f"Removed {initial_count - len(df)} rows with NaN values")
            
            # Clean sentiment values by stripping whitespace
            df['Sentiment'] = df['Sentiment'].str.strip()
            
            # Print unique sentiment values to understand what we're working with
            print("Unique sentiment values in the dataset:")
            print(df['Sentiment'].value_counts())
            
            # Create a mapping from text sentiment to binary labels
            sentiment_mapping = {
                'Positive': 1,
                'Negative': 0,
                'Neutral': 0,  # Map neutral to negative
            }
            
            # Filter for rows with known sentiment values
            df = df[df['Sentiment'].isin(sentiment_mapping.keys())]
            print(f"After filtering, using {len(df)} rows with valid sentiment values")
            
            if len(df) == 0:
                raise ValueError("No valid sentiment values found in dataset")
            
            # Map sentiment values to binary labels
            df['label'] = df['Sentiment'].map(sentiment_mapping)
            texts = df['Text'].tolist()
            
            # Check class distribution
            print("Class distribution:")
            print(df['label'].value_counts())
            
        else:
            # Fall back to the original expected structure
            print("Attempting to use original dataset format")
            df.columns = ['label', 'id', 'date', 'query', 'user', 'text']
            
            # Drop rows with NaN values
            df = df.dropna(subset=['label', 'text'])
            
            # Map labels to binary (0: Negative, 1: Positive)
            label_mapping = {0: 0, 4: 1}  # Adjust as needed for your dataset
            df = df[df['label'].isin(label_mapping.keys())]
            df['label'] = df['label'].map(label_mapping)
            texts = df['text'].tolist()
        
        # Check if we have enough data left
        if len(df) < 10:
            raise ValueError(f"Not enough valid samples for training: only {len(df)} samples after filtering.")
        
        # Tokenize the text data
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if model_type == "bert" else DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Save the processed data for reuse
        torch.save({'encodings': encodings, 'labels': torch.tensor(df['label'].values, dtype=torch.long)}, processed_data_path)
        print(f"Processed data saved to {processed_data_path}")
        
        # Call the existing prepare_data function to split and balance the data
        return prepare_data(processed_data_path, device, model_type)
    
    except Exception as e:
        print(f"Error preparing data with new dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def prepare_twitter_dataset(raw_data_path, processed_data_path, device, model_type=MODEL_TYPE):
    """Load and prepare the Twitter dataset for training."""
    print(f"Loading Twitter dataset from {raw_data_path}...")
    try:
        # Load the Twitter dataset with correct format handling
        df = pd.read_csv(raw_data_path, encoding='latin-1', header=None)
        df.columns = ['label', 'id', 'date', 'query', 'user', 'text']
        
        # Print dataset information
        print(f"Loaded Twitter dataset with {len(df)} tweets")
        
        # Map Twitter sentiment labels to maintain 5-class structure
        # Original Twitter data uses 0 (negative) and 4 (positive)
        # We'll map these to the correct 5-class sentiment scale
        label_mapping = {0: 0, 4: 4}  # Keep original values instead of mapping to binary
        df = df[df['label'].isin(label_mapping.keys())]
        df['label'] = df['label'].map(label_mapping)

        # Report class distribution
        print("Twitter dataset class distribution after mapping:")
        class_distribution = df['label'].value_counts()
        print(class_distribution)

        # Optional: Add synthetic examples for missing classes to improve model robustness
        # This helps the model learn the full spectrum of sentiment
        if AUGMENT_MISSING_CLASSES:
            print("Adding synthetic examples for missing classes...")
            
            # Sample texts from existing classes
            neg_texts = df[df['label'] == 0]['text'].sample(n=min(1000, len(df[df['label'] == 0]))).tolist()
            pos_texts = df[df['label'] == 4]['text'].sample(n=min(1000, len(df[df['label'] == 4]))).tolist()
            
            # Create synthetic examples for missing classes
            synthetic_examples = []
            
            # Class 1 (Somewhat Negative) - modify negative examples
            for text in neg_texts[:500]:
                synthetic_examples.append({
                    'text': text + " although there are some minor positive aspects.",
                    'label': 1
                })
            
            # Class 2 (Neutral) - combine aspects of both positive and negative
            for i in range(min(500, len(neg_texts), len(pos_texts))):
                synthetic_examples.append({
                    'text': f"On one hand, {neg_texts[i].split('.')[0]}. On the other hand, {pos_texts[i].split('.')[0]}.",
                    'label': 2
                })
            
            # Class 3 (Somewhat Positive) - modify positive examples
            for text in pos_texts[:500]:
                synthetic_examples.append({
                    'text': text + " though there are some minor concerns.",
                    'label': 3
                })
            
            # Add synthetic examples to the dataframe
            synthetic_df = pd.DataFrame(synthetic_examples)
            df = pd.concat([df, synthetic_df], ignore_index=True)
            
            # Shuffle the dataframe
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Report updated class distribution
            print("Class distribution after augmentation:")
            print(df['label'].value_counts())
        
        # Check class distribution
        print("Twitter dataset class distribution:")
        class_distribution = df['label'].value_counts()
        print(class_distribution)
        
        # Check if we need to sample for faster training
        if USE_SUBSET and len(df) > SUBSET_SIZE:
            # Balanced sampling for training set
            df_neg = df[df['label'] == 0]
            df_pos = df[df['label'] == 4]
            
            # Sample equally from each class
            samples_per_class = min(SUBSET_SIZE // 2, len(df_neg), len(df_pos))
            df_neg_sampled = df_neg.sample(n=samples_per_class, random_state=42)
            df_pos_sampled = df_pos.sample(n=samples_per_class, random_state=42)
            
            # Combine the samples
            df = pd.concat([df_neg_sampled, df_pos_sampled]).sample(frac=1, random_state=42)
            print(f"Sampled dataset to {len(df)} tweets for faster training")
        
        # Tokenize the text data in batches to avoid memory issues
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') if model_type == "distilbert" else BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Process in batches to handle large datasets
        batch_size = 10000
        all_input_ids = []
        all_attention_mask = []
        
        for i in range(0, len(df), batch_size):
            end_idx = min(i + batch_size, len(df))
            batch_texts = df['text'][i:end_idx].tolist()
            
            print(f"Tokenizing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            # Tokenize the batch
            batch_encodings = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=128,  # Twitter messages are short, so we can use a smaller max_length
                return_tensors='pt'
            )
            
            # Store batch results
            all_input_ids.append(batch_encodings['input_ids'])
            all_attention_mask.append(batch_encodings['attention_mask'])
            
            # Free memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Combine batches
        input_ids = torch.cat(all_input_ids)
        attention_mask = torch.cat(all_attention_mask)
        
        # Create the full encodings dictionary
        encodings = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Convert labels to tensor
        labels = torch.tensor(df['label'].values, dtype=torch.long)
        
        # Save the processed data for future use
        torch.save({'encodings': encodings, 'labels': labels}, processed_data_path)
        print(f"Processed Twitter data saved to {processed_data_path}")
        
        # Call prepare_data to handle the train/validation split
        return prepare_data(processed_data_path, device, model_type)
        
    except Exception as e:
        print(f"Error preparing Twitter dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

class EMA:
    """
    Exponential Moving Average for model weights
    Helps produce more stable models, especially in the final epochs
    """
    def __init__(self, model, decay=EMA_DECAY):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class Lookahead:
    """
    Lookahead optimizer implementation
    Improves convergence by looking ahead in parameter space
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast_p.data)
                param_state["slow_param"].copy_(fast_p.data)
            slow = param_state["slow_param"]
            slow.add_(self.alpha * (fast_p.data - slow))
            fast_p.data.copy_(slow)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] = (group["counter"] + 1) % self.k
        return loss
    
    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }
    
    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)

def create_optimizers(model, total_steps):
    """
    Create advanced optimizers with layerwise learning rates and weight decay
    """
    # Group parameters for different learning rates and weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    
    # Parameters for layerwise learning rates
    if model.model_type == "distilbert":
        # Get the layers from DistilBERT
        transformer_layers = [(f"transformer.layer.{i}.", i) for i in range(6)]
        
        # Parameters with grouped learning rates and weight decay
        optimizer_grouped_parameters = []
        
        # Add classifier parameters first - highest learning rate
        classifier_params = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(x in n for x in ['classifier', 'pre_classifier', 'middle', 'layer_norm']) 
                          and not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': 1e-4
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(x in n for x in ['classifier', 'pre_classifier', 'middle', 'layer_norm']) 
                          and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': 1e-4
            }
        ]
        optimizer_grouped_parameters.extend(classifier_params)
        
        # Add layerwise learning rates for transformer layers - higher layers get higher rates
        for layer_name, layer_idx in transformer_layers:
            # Higher layers get higher learning rates (if unfrozen)
            layer_lr = 3e-5 * (LAYERWISE_LR_DECAY ** (5 - layer_idx))
            
            # Only add if layer is unfrozen (layer 4 and 5)
            if layer_idx >= 4:
                optimizer_grouped_parameters.extend([
                    {
                        'params': [p for n, p in model.named_parameters() 
                                if layer_name in n and not any(nd in n for nd in no_decay)],
                        'weight_decay': 0.01,
                        'lr': layer_lr,
                    },
                    {
                        'params': [p for n, p in model.named_parameters() 
                                if layer_name in n and any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0,
                        'lr': layer_lr,
                    }
                ])
    else:
        # For BERT, create similar layerwise groups
        # This code would be similar to DistilBERT but with 12 layers instead of 6
        pass  # Not used since we're using DistilBERT
    
    # Create AdamW optimizer with our parameter groups
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    # Apply Lookahead wrapper if enabled
    if USE_LOOKAHEAD:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    
    # Create learning rate scheduler
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    if LR_SCHEDULER == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif LR_SCHEDULER == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:  # "reduce_on_plateau"
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
    
    return optimizer, scheduler

def train_with_advanced_optimization(model, train_dataloader, val_dataloader, config):
    """
    Enhanced training function with state-of-the-art optimization techniques:
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Learning rate scheduling (One-cycle policy)
    - Gradient clipping
    - Stochastic Weight Averaging (SWA)
    - Exponential Moving Average (EMA)
    - Sharpness-Aware Minimization (SAM)
    - Lookahead optimizer wrapper
    - AdamW with weight decay and bias correction
    - Early stopping with patience
    - Focal Loss for imbalanced datasets
    """
    # Set all seeds for reproducibility
    set_seed(config.get("seed", 42))
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize EMA model
    ema = EMA(model, decay=config.get("ema_decay", 0.999)) if config.get("use_ema", True) else None
    
    # Initialize optimizer with weight decay
    # Different parameter groups for different learning rates
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.get("weight_decay", 0.01),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Base optimizer
    base_optimizer = torch.optim.AdamW
    optimizer_kwargs = {
        "lr": config.get("learning_rate", 2e-5),
        "eps": config.get("adam_epsilon", 1e-8),
        "betas": (config.get("beta1", 0.9), config.get("beta2", 0.999)),
    }
    
    # Apply SAM optimizer if enabled
    if config.get("use_sam", False):
        optimizer = SAM(optimizer_grouped_parameters, base_optimizer, rho=0.05, **optimizer_kwargs)
    else:
        optimizer = base_optimizer(optimizer_grouped_parameters, **optimizer_kwargs)
    
    # Apply Lookahead if enabled
    if config.get("use_lookahead", False):
        optimizer = Lookahead(optimizer, k=config.get("lookahead_k", 5), alpha=config.get("lookahead_alpha", 0.5))
    
    # Setup SWA if enabled
    swa_model = None
    if config.get("use_swa", False):
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer, 
            swa_lr=config.get("swa_lr", 1e-6), 
            anneal_epochs=config.get("swa_anneal_epochs", 5)
        )
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * config.get("num_epochs", 5) // config.get("gradient_accumulation_steps", 1)
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    
    if config.get("scheduler", "cosine") == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps,
            eta_min=config.get("min_lr", 1e-7)
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=config.get("scheduler_factor", 0.5),
            patience=config.get("scheduler_patience", 2),
            min_lr=config.get("min_lr", 1e-7)
        )
    
    # Loss function - regular cross entropy or focal loss for imbalanced data
    if config.get("use_focal_loss", False):
        # Calculate class weights if needed
        if train_dataloader.dataset.num_classes > 0:
            class_counts = torch.zeros(train_dataloader.dataset.num_classes)
            for batch in train_dataloader:
                labels = batch[1]
                for label in labels:
                    class_counts[label] += 1
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * train_dataloader.dataset.num_classes
            criterion = FocalLoss(gamma=config.get("focal_gamma", 2.0), alpha=class_weights.to(device))
        else:
            criterion = FocalLoss(gamma=config.get("focal_gamma", 2.0))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Early stopping setup
    best_val_metric = float('inf') if config.get("metric_mode", "min") == "min" else float('-inf')
    early_stopping_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(config.get("num_epochs", 5)):
        model.train()
        train_loss = 0.0
        
        # Progress bar for training
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.get('num_epochs', 5)}")
        
        for step, batch in enumerate(progress_bar):
            # Unpack the batch
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 2 else None
            labels = batch[-1].to(device)
            
            # Determine if this is a SAM step
            is_sam_step = config.get("use_sam", False)
            
            # Gradient accumulation logic
            accumulation_steps = config.get("gradient_accumulation_steps", 1)
            should_optimize = (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader)
            
            # Forward pass with mixed precision
            with autocast(enabled=config.get("use_amp", True)):
                if is_sam_step:
                    # For SAM, we need a closure
                    def closure():
                        optimizer.zero_grad()
                        outputs = model(input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels) / accumulation_steps
                        scaler.scale(loss).backward()
                        return loss
                    
                    # SAM first step
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels) / accumulation_steps
                    scaler.scale(loss).backward()
                    
                    if should_optimize:
                        # SAM first step - perturb weights
                        optimizer.first_step(zero_grad=True)
                        
                        # SAM second step - compute gradient at perturbed weights
                        with autocast(enabled=config.get("use_amp", True)):
                            outputs = model(input_ids, attention_mask=attention_mask)
                            loss = criterion(outputs.logits, labels) / accumulation_steps
                            scaler.scale(loss).backward()
                        
                        # SAM second step - update weights
                        optimizer.second_step(zero_grad=True)
                        
                        # Update learning rate
                        if config.get("scheduler", "cosine") == "cosine":
                            scheduler.step()
                else:
                    # Standard training step
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels) / accumulation_steps
                    
                    # Backward pass with mixed precision
                    scaler.scale(loss).backward()
                    
                    if should_optimize:
                        # Gradient clipping
                        if config.get("max_grad_norm", 1.0) > 0:
                            scaler.unscale_(optimizer)
                            clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                        
                        # Optimizer step with gradient scaling
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        # Update learning rate
                        if config.get("scheduler", "cosine") == "cosine":
                            scheduler.step()
            
            train_loss += loss.item() * accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item() * accumulation_steps,
                "lr": optimizer.param_groups[0]["lr"]
            })
            
            # EMA update
            if ema is not None and should_optimize:
                ema.update()
            
            # Update SWA model if it's time
            if swa_model is not None and epoch >= config.get("swa_start_epoch", 3) and should_optimize:
                swa_model.update_parameters(model)
                swa_scheduler.step()
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Apply EMA shadow weights for validation if enabled
        if ema is not None:
            ema.apply_shadow()
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 2 else None
                labels = batch[-1].to(device)
                
                with autocast(enabled=config.get("use_amp", True)):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                
                # Convert logits to predictions
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Calculate validation metrics
        val_accuracy = torchmetrics.functional.accuracy(torch.tensor(all_preds), torch.tensor(all_labels))
        val_f1 = F1Score(num_classes=2, average="macro")(torch.tensor(all_preds), torch.tensor(all_labels))
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{config.get('num_epochs', 5)}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"Val F1 Score: {val_f1:.4f}")
        
        # Restore original weights if using EMA
        if ema is not None:
            ema.restore()
        
        # Update learning rate for ReduceLROnPlateau
        if config.get("scheduler", "cosine") != "cosine":
            scheduler.step(avg_val_loss)
        
        # Check if we should save the model (best validation metric)
        val_metric = avg_val_loss if config.get("metric_mode", "min") == "min" else val_f1
        improved = (val_metric < best_val_metric) if config.get("metric_mode", "min") == "min" else (val_metric > best_val_metric)
        
        if improved:
            best_val_metric = val_metric
            best_model_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
            }
            
            # Save EMA state if used
            if ema is not None:
                best_model_state["ema"] = ema.shadow
            
            # Save SWA state if used
            if swa_model is not None:
                best_model_state["swa_model"] = swa_model.state_dict()
            
            # Save the best model
            torch.save(best_model_state, config.get("model_save_path", "best_model.pt"))
            logger.info(f"New best model saved with val_metric: {val_metric:.4f}")
            
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(f"No improvement in val_metric. Counter: {early_stopping_counter}/{config.get('early_stopping_patience', 3)}")
        
        # Early stopping
        if early_stopping_counter >= config.get("early_stopping_patience", 3):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Update model with SWA weights at the end if enabled
    if swa_model is not None:
        # Update batch norm statistics
        logger.info("Updating batch norm statistics for SWA model")
        with torch.no_grad():
            for batch in train_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 2 else None
                swa_model(input_ids, attention_mask=attention_mask)
        
        # Save SWA model
        torch.save({
            "model": swa_model.state_dict(),
            "epoch": config.get("num_epochs", 5),
        }, config.get("swa_model_save_path", "swa_model.pt"))
        logger.info("SWA model saved")
    
    # Load the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state["model"])
        logger.info("Loaded best model weights")
    
    return model, best_val_metric

def train_model(model, train_data, val_data, device, batch_size=BATCH_SIZE, epochs=EPOCHS, model_type=MODEL_TYPE):
    """
    Train the model with batching and progress tracking.
    Optimized for large datasets with gradient accumulation and early stopping.
    """
    try:
        # Unpack data
        train_input_ids, train_attn_mask, train_labels = train_data
        val_input_ids, val_attn_mask, val_labels = val_data
        
        # Create data loaders for batching
        train_dataset = TensorDataset(train_input_ids, train_attn_mask, train_labels)
        val_dataset = TensorDataset(val_input_ids, val_attn_mask, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)  # Larger batch for validation
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer with learning rate decay
        if model_type == "bert":
            # Use different learning rates for different parts of the model
            bert_params = list(model.bert.parameters())
            classifier_params = list(model.classifier.parameters())
            
            optimizer = optim.AdamW([
                {'params': bert_params, 'lr': 2e-5},
                {'params': classifier_params, 'lr': 1e-4}
            ], weight_decay=0.01)
        else:
            # For DistilBERT, we'll selectively unfreeze some layers for fine-tuning
            # This gives better performance on the large dataset
            for param in model.bert.parameters():
                param.requires_grad = False
                
            # Only unfreeze the last transformer layers for fine-tuning
            for layer in model.bert.transformer.layer[-2:]:  # Unfreeze last 2 layers
                for param in layer.parameters():
                    param.requires_grad = True
            
            optimizer = optim.AdamW([
                {'params': [p for p in model.parameters() if p.requires_grad], 'lr': 3e-5}
            ], weight_decay=0.01)
        
        # Add learning rate scheduler - cosine schedule with warmup
        total_steps = len(train_loader) * epochs // GRADIENT_ACCUMULATION_STEPS
        warmup_steps = int(total_steps * 0.1)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.03, 0.5 * (1.0 + np.cos(np.pi * progress)))  # Min LR = 3% of max
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Track best model and early stopping
        best_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            
            # Training with gradient accumulation
            optimizer.zero_grad()  # Zero gradients at the beginning of epoch
            
            for i, (input_ids, attn_mask, targets) in enumerate(train_loader):
                # Forward pass
                outputs = model(input_ids, attn_mask)
                loss = criterion(outputs, targets)
                
                # Scale the loss by accumulation steps
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                # Update running loss
                running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                
                # Perform optimizer step and reset gradients after accumulation steps
                if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Print statistics
                    if (i + 1) // GRADIENT_ACCUMULATION_STEPS % 10 == 0:
                        avg_loss = running_loss / (10 * GRADIENT_ACCUMULATION_STEPS)
                        print(f'Epoch {epoch+1}, Step {(i+1)//GRADIENT_ACCUMULATION_STEPS}: ' + 
                              f'Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}')
                        running_loss = 0.0
                
                # Clear GPU cache periodically
                if device.type == 'cuda' and i % 100 == 99:
                    torch.cuda.empty_cache()
            
            # Handle any remaining gradients at the end of epoch
            if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
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
                    val_loss += loss.item() * targets.size(0)
            
            # Calculate metrics
            epoch_time = time.time() - start_time
            accuracy = 100 * correct / total
            avg_val_loss = val_loss / total
            
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
            
            # Save best model and check for early stopping
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs. Best accuracy: {best_accuracy:.2f}%")
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Save checkpoint after each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy,
            }, f'checkpoint_epoch_{epoch+1}.pt')
            
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"Restored best model with validation accuracy: {best_accuracy:.2f}%")
        
        logger.info(f"Model training completed with accuracy: {best_accuracy:.2f}%")
        return model, best_accuracy
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def train_model_optimized(model, train_data, val_data, device, batch_size=BATCH_SIZE, epochs=EPOCHS, model_type=MODEL_TYPE):
    """
    Advanced model training with state-of-the-art optimization techniques:
    - Mixed precision training (FP16)
    - Gradient accumulation for larger effective batch sizes
    - Learning rate scheduling with warmup and cosine decay
    - Advanced memory management with fragmentation reduction
    - Checkpoint saving at regular intervals
    - Live progress visualization with ETA
    - Detailed performance metrics and class-specific accuracy reporting
    """
    try:
        # Ensure checkpoint directory exists
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Unpack data
        train_input_ids, train_attn_mask, train_labels = train_data
        val_input_ids, val_attn_mask, val_labels = val_data
        
        # Create data loaders for batching with memory-optimized settings
        train_dataset = TensorDataset(train_input_ids, train_attn_mask, train_labels)
        val_dataset = TensorDataset(val_input_ids, val_attn_mask, val_labels)
        
        # Use pin_memory for faster GPU transfer and persistent workers for efficiency
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=2 if device.type == 'cuda' else 0,
            persistent_workers=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size*2,  # Larger batch for validation
            shuffle=False,
            pin_memory=True,
            num_workers=2 if device.type == 'cuda' else 0,
            persistent_workers=True if device.type == 'cuda' else False
        )
        
        # Calculate total steps for optimization scheduling
        total_steps = len(train_loader) * epochs // GRADIENT_ACCUMULATION_STEPS
        
        # Initialize mixed precision training
        scaler = GradScaler(enabled=MIXED_PRECISION and device.type == 'cuda')
        
        # Initialize EMA for model stabilization
        ema = EMA(model, decay=EMA_DECAY)
        
        # Define loss function - Focal Loss for better handling of class imbalance
        if USE_FOCAL_LOSS:
            # Calculate class weights based on training data distribution
            class_counts = torch.bincount(train_labels)
            class_weights = 1.0 / (class_counts.float() / class_counts.sum())
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, alpha=class_weights.to(device))
            print(f"Using Focal Loss with class weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Set up layerwise learning rates and weight decay
        optimizer, scheduler = create_optimizers(model, total_steps)
        
        # Track metrics for reporting
        training_stats = []
        best_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        global_step = 0
        
        # Calculate training time estimate
        total_training_steps = epochs * len(train_loader)
        estimated_time_per_step = 0
        start_time = time.time()
        
        # Training loop with progress visualization
        print(f"Starting training for {epochs} epochs ({total_training_steps} steps)...")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            epoch_start = time.time()
            
            # Initialize progress bar with estimated time
            progress_bar = tqdm(
                enumerate(train_loader), 
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="batch"
            )
            
            # Zero gradients at the beginning of epoch
            optimizer.zero_grad()
            
            for i, (input_ids, attn_mask, targets) in progress_bar:
                step_start = time.time()
                
                # Move data to device if not already there
                if input_ids.device != device:
                    input_ids = input_ids.to(device, non_blocking=True)
                    attn_mask = attn_mask.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                
                # Mixed precision forward pass
                with autocast(enabled=MIXED_PRECISION and device.type == 'cuda'):
                    outputs = model(input_ids, attn_mask)
                    loss = criterion(outputs, targets)
                    
                    # Scale the loss by accumulation steps
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass with gradient scaling for mixed precision
                scaler.scale(loss).backward()
                
                # Update running loss (scaled back for reporting)
                running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                
                # Perform optimizer step and reset gradients after accumulation steps
                if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    # Gradient clipping to prevent exploding gradients
                    if MIXED_PRECISION and device.type == 'cuda':
                        scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                    
                    # Update weights with gradient scaling for mixed precision
                    if MIXED_PRECISION and device.type == 'cuda':
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Update learning rate
                    scheduler.step()
                    
                    # Update EMA model
                    ema.update()
                    
                    # Reset gradients
                    optimizer.zero_grad()
                    
                    # Increment global step
                    global_step += 1
                    
                    # Save checkpoint at regular intervals
                    if global_step % CHECKPOINT_INTERVAL == 0:
                        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_step_{global_step}.pt')
                        torch.save({
                            'step': global_step,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': running_loss / (i+1),
                        }, checkpoint_path)
                        print(f"Checkpoint saved to {checkpoint_path}")
                
                # Calculate time estimates
                step_time = time.time() - step_start
                estimated_time_per_step = estimated_time_per_step * 0.95 + step_time * 0.05 if estimated_time_per_step > 0 else step_time
                steps_remaining = total_training_steps - (epoch * len(train_loader) + i + 1)
                eta_seconds = steps_remaining * estimated_time_per_step
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # Update progress bar with current metrics
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                avg_loss = running_loss / (i+1)
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'ETA': eta_str
                })
                
                # Clear GPU cache periodically to reduce fragmentation
                if device.type == 'cuda' and i % 200 == 199:
                    torch.cuda.empty_cache()
            
            # Epoch complete, calculate time taken
            epoch_time = time.time() - epoch_start
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_predictions = []
            all_targets = []
            
            # Apply EMA weights for validation
            ema.apply_shadow()
            
            print("\nValidating model...")
            # Progress bar for validation
            val_progress = tqdm(val_loader, desc="Validation")
            
            with torch.no_grad():
                for input_ids, attn_mask, targets in val_progress:
                    # Move data to device if needed
                    if input_ids.device != device:
                        input_ids = input_ids.to(device, non_blocking=True)
                        attn_mask = attn_mask.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                    
                    # Mixed precision inference
                    with autocast(enabled=MIXED_PRECISION and device.type == 'cuda'):
                        outputs = model(input_ids, attn_mask)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * targets.size(0)
                    
                    # Store predictions and targets for detailed metrics
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # Restore original model weights
            ema.restore()
            
            # Calculate metrics
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            accuracy = 100 * np.mean(all_predictions == all_targets)
            avg_val_loss = val_loss / len(all_targets)
            
            # Calculate per-class metrics
            class_accuracies = {}
            class_f1_scores = {}
            
            for class_id in range(2):  # Binary classification
                class_mask = (all_targets == class_id)
                if np.sum(class_mask) > 0:
                    class_correct = np.sum((all_predictions == class_id) & class_mask)
                    class_total = np.sum(class_mask)
                    class_accuracy = 100 * class_correct / class_total
                    class_accuracies[class_id] = class_accuracy
                    
                    # Calculate precision and recall for F1 score
                    true_positives = np.sum((all_predictions == class_id) & (all_targets == class_id))
                    predicted_positives = np.sum(all_predictions == class_id)
                    actual_positives = np.sum(all_targets == class_id)
                    
                    precision = true_positives / max(predicted_positives, 1)
                    recall = true_positives / max(actual_positives, 1)
                    
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                    class_f1_scores[class_id] = f1
            
            # Print detailed metrics
            print(f'\n--- Epoch {epoch + 1}/{epochs} Summary ---')
            print(f'Time: {epoch_time:.2f}s')
            print(f'Train Loss: {running_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Validation Accuracy: {accuracy:.2f}%')
            
            # Display confusion matrix
            from torchmetrics.classification import ConfusionMatrix
            cm = ConfusionMatrix(num_classes=2)(torch.tensor(all_predictions), torch.tensor(all_targets))
            print("\nConfusion Matrix:")
            print(cm)
            
            # Display per-class metrics
            print("\nPer-class Performance:")
            for class_id in range(2):
                if class_id in class_accuracies:
                    print(f"  Class {class_id}: Accuracy={class_accuracies[class_id]:.2f}%, F1-Score={class_f1_scores[class_id]:.4f}")
            
            # Save statistics for this epoch
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': running_loss/len(train_loader),
                'val_loss': avg_val_loss,
                'accuracy': accuracy,
                'class_accuracies': class_accuracies,
                'class_f1_scores': class_f1_scores,
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
            })
            
            # Save best model and check for early stopping
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': accuracy,
                    'val_loss': avg_val_loss,
                    'ema_state': ema.shadow,
                }
                
                # Save best model checkpoint
                best_model_path = os.path.join(CHECKPOINT_DIR, f'best_model_{accuracy:.2f}.pt')
                torch.save(best_model_state, best_model_path)
                print(f"\n✅ New best model saved with accuracy: {best_accuracy:.2f}% to {best_model_path}")
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                print(f"\nNo improvement for {patience_counter} epochs. Best accuracy: {best_accuracy:.2f}%")
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save checkpoint after each epoch
            epoch_checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy,
                'ema_state': ema.shadow,
            }, epoch_checkpoint_path)
            print(f"Epoch checkpoint saved to {epoch_checkpoint_path}")
            
            # Clear GPU cache at the end of epoch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {datetime.timedelta(seconds=int(total_time))}")
        print(f"Best validation accuracy: {best_accuracy:.2f}%")
        
        # Visualize training results
        epochs_range = list(range(1, len(training_stats) + 1))
        plt.figure(figsize=(12, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, [stats['train_loss'] for stats in training_stats], 'b-o', label='Training')
        plt.plot(epochs_range, [stats['val_loss'] for stats in training_stats], 'r-o', label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, [stats['accuracy'] for stats in training_stats], 'g-o')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        # Plot class-specific accuracy
        plt.subplot(2, 2, 3)
        for class_id in range(2):
            values = [stats['class_accuracies'].get(class_id, 0) for stats in training_stats]
            plt.plot(epochs_range, values, marker='o', label=f'Class {class_id}')
        plt.title('Per-Class Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, [stats['learning_rate'] for stats in training_stats], 'k-o')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_metrics.png'))
        print(f"Training metrics visualization saved to {os.path.join(CHECKPOINT_DIR, 'training_metrics.png')}")
        
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state['model_state_dict'])
            print(f"Restored best model with validation accuracy: {best_accuracy:.2f}%")
        
        return model, best_accuracy
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def evaluate_model(model, val_dataloader, device):
    """
    Evaluate the model on validation data using PyTorch metrics instead of scikit-learn
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    # Initialize metrics
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro").to(device)
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=2, average="macro").to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=2, average="macro").to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=2).to(device)
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Update metrics
            accuracy_metric(preds, labels)
            f1_metric(preds, labels)
            precision_metric(preds, labels)
            recall_metric(preds, labels)
            confusion_matrix(preds, labels)
            
            # Store predictions and labels for later analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute final metrics
    accuracy = accuracy_metric.compute().item()
    f1 = f1_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    cm = confusion_matrix.compute().cpu().numpy()
    
    # Create classification report similar to sklearn's but using our computed metrics
    report = {
        "accuracy": accuracy,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "confusion_matrix": cm,
        "loss": total_loss / len(val_dataloader)
    }
    
    # Add per-class metrics
    for class_idx in range(2):
        true_positives = cm[class_idx, class_idx]
        false_positives = cm[:, class_idx].sum() - true_positives
        false_negatives = cm[class_idx, :].sum() - true_positives
        
        class_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        class_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        class_name = "Positive" if class_idx == 1 else "Negative"
        report[f"{class_name}_precision"] = class_precision
        report[f"{class_name}_recall"] = class_recall
        report[f"{class_name}_f1"] = class_f1
    
    return report, all_preds, all_labels

from src.utils.sentiment_model_saver import SentimentAnalysisModelWrapper, save_model as save_model_util

def save_model(model, model_type=MODEL_TYPE, output_path='model.pkl'):
    """Save the trained model for application use."""
    print(f"Saving model to {output_path}...")
    
    # Move model to CPU for saving
    model.cpu()
    
    # Create the wrapper model using the utility from sentiment_model_saver.py
    wrapper_model = SentimentAnalysisModelWrapper(model, model_type)
    
    # Save using joblib
    try:
        save_model_util(wrapper_model, output_path)
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
    # Check if GPU is available and ask for confirmation to use CPU if not available
    gpu_available = verify_gpu()
    
    if gpu_available:
        device = torch.device('cuda')
        print(f"Using device: {device}")
        # Reset GPU state
        reset_gpu()
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
        print("WARNING: Training on CPU will be significantly slower. This may take hours or days to complete.")
    
    # Process the 1.6 million Twitter dataset
    print(f"Using large Twitter dataset with 1.6 million tweets for fine-tuning")
    twitter_data_path = LARGE_TWITTER_DATASET
    processed_twitter_data_path = 'processed_twitter_data.pt'

    # Prepare data using the Twitter dataset
    train_input_ids, train_attn_mask, val_input_ids, val_attn_mask, train_labels, val_labels = prepare_twitter_dataset(
        twitter_data_path, processed_twitter_data_path, device, model_type=MODEL_TYPE
    )
    
    if train_input_ids is None:
        print("Failed to prepare Twitter data. Exiting.")
        sys.exit(1)

    # Initialize model with improved architecture for better fine-tuning
    try:
        # Create an enhanced model for fine-tuning
        model = SentimentClassifier(model_type=MODEL_TYPE).to(device)
        print(f"Created enhanced {MODEL_TYPE.upper()} model with fine-tuning optimizations and moved to {device}")

        # Fine-tune model with optimized parameters for the large Twitter dataset
        trained_model, accuracy = train_model(
            model, 
            train_data=(train_input_ids, train_attn_mask, train_labels), 
            val_data=(val_input_ids, val_attn_mask, val_labels),
            device=device,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            model_type=MODEL_TYPE
        )

        # Save model with higher threshold for production quality
        if accuracy > SAVE_THRESHOLD:
            # Save with a descriptive name that indicates it's trained on Twitter data
            model_output_path = os.path.join('models', f'twitter_sentiment_{MODEL_TYPE}_{accuracy:.1f}.pkl')
            os.makedirs('models', exist_ok=True)  # Ensure models directory exists
            
            save_model(trained_model, model_type=MODEL_TYPE, output_path=model_output_path)
            
            # Also save to the standard location for backward compatibility
            save_model(trained_model, model_type=MODEL_TYPE)
            
            print(f"Model with {accuracy:.2f}% accuracy saved successfully")
        else:
            print(f"Model accuracy ({accuracy:.2f}%) is below target of {SAVE_THRESHOLD}%. Model not saved.")

    except Exception as e:
        print(f"Error during model fine-tuning: {e}")
        import traceback
        traceback.print_exc()

    # Final cleanup
    if gpu_available:
        reset_gpu()

if __name__ == "__main__":
    main()

def train_sentiment_model(config):
    """
    Train a sentiment analysis model with enhanced features for handling stuck training
    
    Args:
        config: Configuration object with all training parameters
    """
    # Set up logging
    setup_logging(config)
    
    # Set random seed for reproducibility
    if config.seed is None:
        config.seed = random.randint(1, 10000)
    set_seed(config.seed)
    logger.info(f"Using random seed: {config.seed}")
    
    # Load and prepare data
    logger.info("===== LOADING DATA =====")
    if os.path.exists(config.processed_data_path):
        logger.info(f"✓ Loading preprocessed data from {config.processed_data_path}")
        train_dataset, val_dataset, label_counts = load_processed_data(config.processed_data_path)
    else:
        logger.error(f"Processed data file not found: {config.processed_data_path}")
        logger.info("Creating processed data from raw training data...")
        # Process raw data and save
        train_dataset, val_dataset, label_counts = prepare_twitter_dataset(
            config.raw_data_path, 
            config.processed_data_path,
            config.max_seq_length,
            config.max_samples
        )
    
    # Log data statistics and perform analysis to detect potential issues
    logger.info("===== DATA ANALYSIS =====")
    total_examples = sum(label_counts.values())
    logger.info(f"Dataset size: {total_examples} examples\n")
    
    # Analyze class distribution
    logger.info("Label distribution:")
    classes = sorted(label_counts.keys())
    max_count = max(label_counts.values())
    min_count = min(label_counts.values()) if label_counts else 0
    imbalance_ratio = max_count / max(min_count, 1)
    
    for label in range(config.num_labels):
        count = label_counts.get(label, 0)
        percentage = (count / total_examples) * 100 if total_examples > 0 else 0
        class_name = get_sentiment_label(label)
        logger.info(f"  Class {label} ({class_name}): {count} examples ({percentage:.2f}%)")
    
    # Check for class imbalance issues
    if imbalance_ratio > 10:
        logger.warning(f"Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
        logger.warning("This may be contributing to the static loss issue.")
    
    # Check for missing classes
    represented_classes = len([c for c in label_counts.values() if c > 0])
    if represented_classes < config.num_labels:
        logger.warning(f"Only {represented_classes} out of {config.num_labels} classes are represented!")
        
        # Apply advanced class balancing if enabled
        if config.enable_advanced_class_balancing and represented_classes < config.num_labels:
            logger.info("===== APPLYING ADVANCED CLASS BALANCING =====")
            train_dataset = balance_dataset_classes(train_dataset, config.num_labels, config.class_balance_strategy)
            # Update label counts after balancing
            label_counts = count_labels(train_dataset)
            logger.info("Updated label distribution after balancing:")
            for label in range(config.num_labels):
                count = label_counts.get(label, 0)
                percentage = (count / len(train_dataset)) * 100 if len(train_dataset) > 0 else 0
                class_name = get_sentiment_label(label)
                logger.info(f"  Class {label} ({class_name}): {count} examples ({percentage:.2f}%)")
    
    # Sequence length analysis 
    seq_lengths = [len(example['input_ids']) for example in train_dataset]
    avg_length = sum(seq_lengths) / len(seq_lengths)
    max_length = max(seq_lengths)
    
    logger.info("\nSequence length statistics:")
    logger.info(f"  Average length: {avg_length:.2f} tokens")
    logger.info(f"  Maximum length: {max_length} tokens")
    logger.info(f"  Configured max length: {config.max_seq_length} tokens")
    
    if avg_length > config.max_seq_length:
        logger.warning(f"Average sequence length ({avg_length:.2f}) exceeds configured max_seq_length ({config.max_seq_length})")
        logger.warning("Consider increasing max_seq_length to retain more context information")
        
    logger.info("===== DATA ANALYSIS COMPLETE =====\n")
    
    # Create data loaders
    train_dataloader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, config.batch_size, shuffle=False)
    
    logger.info(f"✓ Data split: {len(train_dataset)} training, {len(val_dataset)} validation examples")
    
    # Apply advanced data augmentation if enabled
    if config.enable_data_augmentation:
        logger.info("===== APPLYING ADVANCED DATA AUGMENTATION =====")
        train_dataset = apply_data_augmentation(
            train_dataset, 
            config.augmentation_techniques,
            augmentation_factor=config.augmentation_factor,
            target_class_counts=config.target_class_distribution
        )
        # Recreate dataloader with augmented dataset
        train_dataloader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
        logger.info(f"✓ Dataset after augmentation: {len(train_dataset)} training examples")
        
        # Log new class distribution after augmentation
        label_counts = count_labels(train_dataset)
        logger.info("Label distribution after augmentation:")
        for label in range(config.num_labels):
            count = label_counts.get(label, 0)
            percentage = (count / len(train_dataset)) * 100 if len(train_dataset) > 0 else 0
            class_name = get_sentiment_label(label)
            logger.info(f"  Class {label} ({class_name}): {count} examples ({percentage:.2f}%)")
    
    # Initialize model
    logger.info("===== INITIALIZING MODEL =====")
    # ...existing code...

def load_data(config):
    """
    Load preprocessed sentiment analysis data for training.
    
    Args:
        config: Configuration dictionary with data paths and parameters
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Import torch inside the function to ensure it's available
    import torch
    from torch.utils.data import TensorDataset
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine data path from config
    processed_data_path = config.get("processed_data_path", "data/processed/twitter_sentiment.pt")
    
    logger.info(f"Loading data from {processed_data_path}")
    
    try:
        # Add BatchEncoding to safe globals to fix the serialization issue
        import torch.serialization
        from transformers.tokenization_utils_base import BatchEncoding
        torch.serialization.add_safe_globals([BatchEncoding])
        
        # Now load the data with weights_only=False to allow loading BatchEncoding objects
        data = torch.load(processed_data_path, weights_only=False)
        
        encodings, labels = data['encodings'], data['labels']
        
        # Create TensorDataset
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # Split into train and validation sets
        dataset_size = len(labels)
        val_size = int(dataset_size * 0.2)  # 20% for validation
        train_size = dataset_size - val_size
        
        indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets
        train_dataset = TensorDataset(
            input_ids[train_indices],
            attention_mask[train_indices],
            labels[train_indices]
        )
        
        val_dataset = TensorDataset(
            input_ids[val_indices],
            attention_mask[val_indices],
            labels[val_indices]
        )
        
        logger.info(f"Successfully loaded {train_size} training and {val_size} validation examples")
        
        return train_dataset, val_dataset
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return empty datasets as fallback
        logger.warning("Returning empty datasets due to loading error")
        empty_tensor = torch.empty((0, 10), dtype=torch.long)
        empty_labels = torch.empty((0,), dtype=torch.long)
        return (
            TensorDataset(empty_tensor, empty_tensor, empty_labels),
            TensorDataset(empty_tensor, empty_tensor, empty_labels)
        )

def create_model(config):
    """
    Create a sentiment analysis model based on configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        SentimentClassifier: The model
    """
    # Select model type from config or use default
    model_type = config.get("model_type", MODEL_TYPE)
    
    # Create model with specified parameters
    model = SentimentClassifier(
        model_type=model_type,
        hidden_dim=config.get("hidden_dim", 768),
        output_dim=config.get("num_classes", 3)
    )
    
    logger.info(f"Created {model_type} model")
    
    return model