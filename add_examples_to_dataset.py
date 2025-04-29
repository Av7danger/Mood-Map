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

    # New examples to add
    new_examples = [
        {"text": "Dear Gayatri, Hindu Samaj Party is also going to file FIR against you for blasphemous comments and parroting fake narrative against India at international level.", "label": "negative"},
        {"text": "You are a very good person.", "label": "positive"},
        {"text": "FYI: I was in that C spaces for 2hrs after which I left.", "label": "neutral"},
        {"text": "The organization has decided to take legal action against individuals spreading misinformation about the community.", "label": "negative"},
        {"text": "The team has done an excellent job in organizing the event, ensuring everything went smoothly.", "label": "positive"},
        {"text": "The discussion in the meeting was balanced, with both sides presenting their arguments clearly.", "label": "neutral"},
        {"text": "#Feminism and #Equality don't work in a practical world. Who had the bright idea to showcase feminine power on such a sombre occasion and disrespect the departed soul. It also underlines the unsuitability of women in Armed Forces, where physical capabilities are critical.", "label": "negative"},
        {"text": "Before me, 7 to 8 people enjoyed Zip Ride, but the operator didn't chant 'Allah hu Akbar.' It was only after the first shots were fired that he did so. Kashmiri Muslims were never innocents. Kashmiriyat my foot!!", "label": "negative"},
        {"text": "Beauty of Kashmiriyat! Father of three, Donnie Smith, 53, died hearing three men laugh at him while stomping him to death. On December 6, 2023, Donnie and other customers were harassed inside of a Columbus, OH Kroger. After being removed by security and not satisfied with just words, they waited outside for Donnie where they ambushed and beat him to death.", "label": "negative"},
        {"text": "Indian Muslim: 'I stand firm for Islam with dignity, without selling my soul to any foreign flag.' Self-Proclaimed Ummah Guardians: 'SubhanAllah! But will you chant for Pakistan and curse your own land?' Indian Muslim: 'Never. My loyalty to Islam doesn't require treason.'", "label": "positive"},
        {"text": "What if India had a news channel or newswire service that was global that broadcast into other countries to get the India viewpoint across? Right now all Indian media caters just to domestic market. There needs to be an Indian equivalent of BBC and of Reuters. Any thoughts?", "label": "neutral"},
        {"text": "The Finance Minister isn't just a messenger at the GST Council, she leads it. If she can't build consensus on basic issues like healthcare affordability, then what exactly is her leadership worth?", "label": "negative"}
    ]

    # Add examples to the dataset
    add_examples_to_dataset(dataset_path, new_examples)