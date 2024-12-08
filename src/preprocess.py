# src/preprocess.py

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """
    Preprocess the input text by:
    - Converting text to lowercase
    - Removing punctuation
    - Tokenizing words
    - Removing stopwords
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize words
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Return the cleaned text
    return ' '.join(filtered_tokens)


if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    # Add test input and print the result
    sample_text = "Hello, this is a sample text to test the preprocess function!"
    processed_text = preprocess_text(sample_text)
    print(f"Original Text: {sample_text}")
    print(f"Processed Text: {processed_text}")
