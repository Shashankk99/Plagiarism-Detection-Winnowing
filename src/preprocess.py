import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

def download_nltk_data():
    required_packages = [
        'punkt',  # Correct tokenizer
        'wordnet',
        'omw-1.4',
        'stopwords',
        'averaged_perceptron_tagger'
    ]
    for package in required_packages:
        try:
            nltk.data.find(f'corpora/{package}')  # Check if resource exists
        except LookupError:
            nltk.download(package)  # Download missing resource

download_nltk_data()

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
