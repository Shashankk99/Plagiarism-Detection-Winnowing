# src/preprocess.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# Ensure NLTK data is downloaded
def download_nltk_data():
    required_packages = [
        'punkt',
        'wordnet',
        'omw-1.4',
        'stopwords',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    for package in required_packages:
        try:
            if package == 'punkt':
                nltk.data.find(f'tokenizers/{package}')
            elif package in ['wordnet', 'omw-1.4', 'stopwords', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            nltk.download(package)

download_nltk_data()

def get_wordnet_pos(nltk_pos_tag):
    """
    Map NLTK POS tags to WordNet POS tags for accurate lemmatization.
    
    Args:
        nltk_pos_tag (str): POS tag from NLTK's pos_tag.
    
    Returns:
        wordnet POS tag or wordnet.NOUN as default.
    """
    if nltk_pos_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_pos_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    """
    Preprocesses the input text by:
    - Converting text to lowercase
    - Removing punctuation
    - Tokenizing words
    - Removing stopwords
    - Lemmatizing tokens based on POS tags
    
    Args:
        text (str): The raw text to preprocess.
        
    Returns:
        str: The preprocessed text as a single string of lemmatized tokens.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize words
    tokens = word_tokenize(text)

    # POS tagging
    pos_tags = pos_tag(tokens)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords and lemmatize tokens based on POS tags
    stop_words = set(stopwords.words('english'))
    lemmatized_tokens = []
    for word, tag in pos_tags:
        if word not in stop_words:
            wordnet_pos = get_wordnet_pos(tag)
            lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
            lemmatized_tokens.append(lemmatized_word)

    # Return the cleaned text
    return ' '.join(lemmatized_tokens)

if __name__ == "__main__":
    # Example usage
    sample_text = "The children are playing with their toys."
    processed_text = preprocess_text(sample_text)
    print(f"Original Text: {sample_text}")
    print(f"Processed Text: {processed_text}")
