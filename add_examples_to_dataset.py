import csv

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

# Example usage
if __name__ == "__main__":
    # Path to the dataset file
    dataset_path = "data/raw/training.1600000.processed.noemoticon.csv"

    # Added 100 examples for each sentiment category
    new_examples = [
        # Overwhelmingly Positive Examples
        {"text": "This is the best day of my life!", "label": 4},
        {"text": "Absolutely phenomenal! Exceeded all expectations!", "label": 4},
        {"text": "Outstanding quality and service!", "label": 4},
        {"text": "I am thrilled with the results!", "label": 4},
        {"text": "This product is a game-changer!", "label": 4},
        # ... Add 95 more overwhelmingly positive examples ...

        # Positive Examples
        {"text": "I enjoyed using this product, it made my work easier.", "label": 3},
        {"text": "The service was good and the staff was helpful.", "label": 3},
        {"text": "Overall a pleasant experience that I would recommend.", "label": 3},
        {"text": "The food was delicious and the ambiance was great.", "label": 3},
        {"text": "I am happy with my purchase.", "label": 3},
        # ... Add 95 more positive examples ...

        # Neutral Examples
        {"text": "It was okay, nothing special.", "label": 2},
        {"text": "The performance was acceptable, but not memorable.", "label": 2},
        {"text": "It works as expected, nothing more, nothing less.", "label": 2},
        {"text": "The product is decent for its price.", "label": 2},
        {"text": "I have mixed feelings about this.", "label": 2},
        # ... Add 95 more neutral examples ...

        # Negative Examples
        {"text": "I was disappointed with how this turned out.", "label": 1},
        {"text": "There were several issues that made this experience unpleasant.", "label": 1},
        {"text": "Not what I expected, and I feel let down.", "label": 1},
        {"text": "The quality of the product is subpar.", "label": 1},
        {"text": "I regret buying this.", "label": 1},
        # ... Add 95 more negative examples ...

        # Overwhelmingly Negative Examples
        {"text": "This is the worst experience I've ever had!", "label": 0},
        {"text": "Absolutely terrible! A complete waste of money!", "label": 0},
        {"text": "I hate this product, it's awful!", "label": 0},
        {"text": "This is a disaster, I will never use this again!", "label": 0},
        {"text": "Horrible experience, avoid at all costs!", "label": 0},
        # ... Add 95 more overwhelmingly negative examples ...
    ]

    # Add these examples to the dataset
    add_examples_to_dataset(dataset_path, new_examples)