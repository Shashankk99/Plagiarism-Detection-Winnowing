# download_nltk_data.py

import nltk

# List of required NLTK data packages
required_packages = [
    'punkt',
    'wordnet',
    'omw-1.4',
    'stopwords',
    'averaged_perceptron_tagger',       # Existing tagger resource
    'averaged_perceptron_tagger_eng'    # Add the language-specific tagger resource
]

def download_packages(packages):
    for package in packages:
        try:
            if package == 'punkt':
                nltk.data.find(f'tokenizers/{package}')
            elif package in ['wordnet', 'omw-1.4', 'stopwords', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
                nltk.data.find(f'corpora/{package}')  # For corpora packages
        except LookupError:
            nltk.download(package)

if __name__ == "__main__":
    download_packages(required_packages)
    print("All necessary NLTK data packages have been downloaded.")
