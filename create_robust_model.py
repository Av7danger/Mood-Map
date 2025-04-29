import pandas as pd

def ensure_text_and_label_columns(data):
    """
    Ensure the DataFrame has proper 'text' and 'label' columns.
    - 'text' column is derived from 'Text'.
    - 'label' column is derived from 'Sentiment' using a mapping.
    """
    print("Current columns:", data.columns.tolist())

    # Ensure 'text' column contains the text data
    if 'text' not in data.columns or data['text'].isna().any():
        print("Ensuring 'text' column contains the text data")
        data['text'] = data['Text']  # Copy from 'Text' to ensure we have the data

    # Ensure 'label' column is properly populated
    if 'label' not in data.columns or data['label'].isna().all():
        print("Creating 'label' column from 'Sentiment'")
        # Create a mapping from sentiment words to numeric values
        sentiment_mapping = {
            'Overwhelmingly Positive': 4,
            'Positive': 3,
            'Neutral': 2,
            'Negative': 1,
            'Overwhelmingly Negative': 0
        }
        data['label'] = data['Sentiment'].map(sentiment_mapping)

    print("Updated DataFrame head:")
    print(data.head())
    print("\nLabel value counts:")
    print(data['label'].value_counts())

    return data