# src/preprocess.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

def download_nltk_resources():
    """
    Downloads necessary NLTK resources if they are not already present.
    """
    required_resources = ['punkt', 'wordnet', 'stopwords']
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            print(f"NLTK resource '{resource}' already downloaded.")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

def preprocess_text(text):
    """
    Preprocesses the input text by performing the following steps:
    1. Lowercasing
    2. Punctuation removal
    3. Tokenization
    4. Stopword removal
    5. Lemmatization

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Ensure required NLTK resources are available
    download_nltk_resources()
    
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize words
    tokens = word_tokenize(text, language='english')  # Corrected language parameter
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back to string
    processed_text = ' '.join(tokens)
    
    return processed_text
