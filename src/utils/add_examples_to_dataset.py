import csv
import random
import re
import nltk
from nltk.corpus import wordnet
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def augment_text(text, techniques=None):
    """
    Apply various text augmentation techniques to create new examples.
    
    Args:
        text (str): The original text to augment
        techniques (list): List of techniques to apply, or None for random selection
    
    Returns:
        str: Augmented text
    """
    if techniques is None:
        # Randomly select 1-2 techniques
        num_techniques = random.randint(1, 2)
        all_techniques = ['synonym_replacement', 'random_insertion', 'random_deletion', 'random_swap']
        techniques = random.sample(all_techniques, num_techniques)
    
    augmented_text = text
    
    if 'synonym_replacement' in techniques:
        # Replace 10-30% of words with synonyms
        words = nltk.word_tokenize(augmented_text)
        num_to_replace = max(1, int(len(words) * random.uniform(0.1, 0.3)))
        indices = random.sample(range(len(words)), min(num_to_replace, len(words)))
        
        for idx in indices:
            word = words[idx]
            # Only replace words longer than 3 characters, not stopwords or special tokens
            if len(word) > 3 and word.isalpha():
                synonyms = get_synonyms(word)
                if synonyms:
                    words[idx] = random.choice(synonyms)
        
        augmented_text = ' '.join(words)
    
    if 'random_insertion' in techniques:
        # Insert 1-3 synonyms of existing words
        words = nltk.word_tokenize(augmented_text)
        num_insertions = random.randint(1, min(3, len(words)))
        
        for _ in range(num_insertions):
            if not words:
                break
            
            word = random.choice(words)
            if len(word) > 3 and word.isalpha():
                synonyms = get_synonyms(word)
                if synonyms:
                    random_synonym = random.choice(synonyms)
                    insert_position = random.randint(0, len(words))
                    words.insert(insert_position, random_synonym)
        
        augmented_text = ' '.join(words)
    
    if 'random_deletion' in techniques:
        # Delete 10-20% of words
        words = nltk.word_tokenize(augmented_text)
        retain_prob = random.uniform(0.8, 0.9)
        
        augmented_words = []
        for word in words:
            if random.random() < retain_prob or len(word) <= 3 or not word.isalpha():
                augmented_words.append(word)
        
        if not augmented_words:  # Ensure we don't delete all words
            augmented_words = words
        
        augmented_text = ' '.join(augmented_words)
    
    if 'random_swap' in techniques:
        # Swap positions of words 1-3 times
        words = nltk.word_tokenize(augmented_text)
        num_swaps = random.randint(1, min(3, len(words)//2))
        
        for _ in range(num_swaps):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        augmented_text = ' '.join(words)
    
    return augmented_text

def add_examples_to_dataset(file_path, examples):
    """
    Add new examples to the dataset.

    Args:
        file_path (str): Path to the dataset file (CSV format).
        examples (list of dict): List of examples to add, where each example is a dictionary
                                 with keys 'text' and 'label'.
    """
    try:
        # Open the file in append mode
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['text', 'label'])

            # Write header if the file is empty
            if file.tell() == 0:
                writer.writeheader()

            # Write the new examples
            for example in examples:
                writer.writerow(example)

        print(f"Successfully added {len(examples)} examples to {file_path}.")
    except Exception as e:
        print(f"Error adding examples to dataset: {e}")

def generate_augmented_dataset(base_examples, num_augmentations=3):
    """
    Generate augmented examples from a base set of examples.
    
    Args:
        base_examples (list): List of dictionaries with 'text' and 'label' keys
        num_augmentations (int): Number of augmented examples to create per original
        
    Returns:
        list: Combined list of original and augmented examples
    """
    all_examples = base_examples.copy()
    
    for example in base_examples:
        original_text = example['text']
        label = example['label']
        
        for _ in range(num_augmentations):
            augmented_text = augment_text(original_text)
            all_examples.append({'text': augmented_text, 'label': label})
    
    print(f"Created augmented dataset with {len(all_examples)} examples from {len(base_examples)} base examples")
    return all_examples

def create_balanced_dataset(file_path, target_per_class=2000):
    """
    Create a balanced dataset with equal representation of each class.
    
    Args:
        file_path (str): Path to the dataset CSV file
        target_per_class (int): Target number of examples per class
        
    Returns:
        str: Path to the new balanced dataset
    """
    import os
    import pandas as pd
    from datetime import datetime
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(dir_path, f"{name}_balanced_{timestamp}{ext}")
    
    try:
        # Read the dataset
        df = pd.read_csv(file_path, encoding='latin-1')
        
        # If the dataset doesn't have text and label columns already named
        if 'text' not in df.columns or 'label' not in df.columns:
            # For Twitter sentiment dataset format
            if len(df.columns) == 6:
                df.columns = ['label', 'id', 'date', 'query', 'user', 'text']
            # Add more format handling as needed
        
        # Get counts per class
        class_counts = df['label'].value_counts()
        print(f"Original class distribution: {class_counts.to_dict()}")
        
        # Identify minority and majority classes
        balanced_df = pd.DataFrame()
        
        for label, count in class_counts.items():
            class_df = df[df['label'] == label]
            
            if count < target_per_class:
                # Need to augment this class
                examples_needed = target_per_class - count
                augmentation_factor = min(5, examples_needed / count)  # Cap at 5x augmentation
                
                # Convert to list of dicts for augmentation
                examples = class_df.to_dict('records')
                
                # Generate augmented examples
                augmented = generate_augmented_dataset(
                    examples, 
                    num_augmentations=int(augmentation_factor)
                )
                
                # If we still need more examples, sample with replacement
                augmented_df = pd.DataFrame(augmented)
                if len(augmented_df) < target_per_class:
                    augmented_df = augmented_df.sample(
                        target_per_class, 
                        replace=True, 
                        random_state=42
                    )
                # If we have too many, sample without replacement
                elif len(augmented_df) > target_per_class:
                    augmented_df = augmented_df.sample(
                        target_per_class, 
                        replace=False, 
                        random_state=42
                    )
                    
                balanced_df = pd.concat([balanced_df, augmented_df])
                
            else:
                # Downsample if we have too many examples
                sampled_df = class_df.sample(target_per_class, random_state=42)
                balanced_df = pd.concat([balanced_df, sampled_df])
        
        # Shuffle the final dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save the balanced dataset
        balanced_df.to_csv(output_path, index=False)
        
        print(f"Created balanced dataset at {output_path}")
        print(f"New class distribution: {balanced_df['label'].value_counts().to_dict()}")
        
        return output_path
        
    except Exception as e:
        print(f"Error creating balanced dataset: {e}")
        import traceback
        traceback.print_exc()
        return file_path  # Return original path on error

# Example usage
if __name__ == "__main__":
    # Path to the dataset file
    dataset_path = "data/raw/sentimentdataset.csv"

    # Example of adding new specialized examples
    domain_specific_examples = [
        # Social media specific sentiments
        {"text": "Just got my new iPhone and I'm absolutely loving it! #blessed #happy", "label": 4},
        {"text": "This new update is pretty good, added some nice features", "label": 3},
        {"text": "The app works fine I guess, some bugs here and there", "label": 2},
        {"text": "This website is so frustrating to use, keeps crashing", "label": 1},
        {"text": "Worst customer service ever!!! Never shopping here again #angry", "label": 0},
        
        # News reaction sentiments
        {"text": "This policy change will revolutionize the industry! Great news!", "label": 4},
        {"text": "The new legislation seems reasonable and addresses some key concerns", "label": 3},
        {"text": "The market was stable today, no major changes to report", "label": 2},
        {"text": "The company missed earnings expectations, disappointing investors", "label": 1},
        {"text": "Devastating economic report shows worst decline in decades", "label": 0},
        
        # Product review specific
        {"text": "5/5 stars! This product exceeded all my expectations", "label": 4},
        {"text": "4 stars - Good product with minor issues", "label": 3},
        {"text": "3/5 - Average performance for the price", "label": 2},
        {"text": "2 stars: Several issues that made the experience frustrating", "label": 1},
        {"text": "1/5 DO NOT BUY! Complete waste of money", "label": 0},
    ]
    
    # First add the examples
    add_examples_to_dataset(dataset_path, domain_specific_examples)
    
    # Then create a balanced version of the dataset
    balanced_dataset_path = create_balanced_dataset(dataset_path, target_per_class=2000)
    
    print(f"Ready to use balanced dataset at: {balanced_dataset_path}")