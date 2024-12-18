# src/preprocess.py

import re
import nltk
import os

nltk.data.path.append(os.path.expanduser("~/.nltk_data"))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def get_wordnet_pos(nltk_pos_tag):
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
    Preprocess text by:
    - Lowercasing
    - Removing punctuation
    - Tokenizing
    - Removing stopwords
    - Lemmatizing

    Complexity:
    - Tokenization: O(n) where n is the number of characters or words.
    - Stopword removal and lemmatization: O(m) where m is number of tokens.
    Overall roughly O(n) for typical text lengths.

    Returns: A cleaned, lemmatized, and space-joined string.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    lemmatized_tokens = []
    for word, tag in pos_tags:
        if word not in stop_words:
            wordnet_pos = get_wordnet_pos(tag)
            lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
            lemmatized_tokens.append(lemmatized_word)
    return ' '.join(lemmatized_tokens)
