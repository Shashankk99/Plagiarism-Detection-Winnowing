
---

```markdown
# Plagiarism Detection Winnowing

## Overview

Plagiarism detection is crucial in maintaining academic integrity and originality in professional content. Traditional methods often rely on lexical overlap, which can fail to identify paraphrased content where the wording differs significantly. This project introduces a **hybrid approach** that combines **lexical fingerprinting** (via rolling hash and winnowing) with **semantic embeddings** from transformer-based models to enhance plagiarism detection capabilities.

## Features

- **Lexical Fingerprinting**: Utilizes rolling hash and winnowing algorithms to identify exact and near-exact text overlaps.
- **Semantic Embeddings**: Employs transformer-based models (e.g., BERT) to capture conceptual and contextual similarities between texts.
- **Hybrid Classification**: Integrates both lexical and semantic similarity scores using an XGBoost classifier to predict plagiarism.
- **Interactive Streamlit Application**: Provides a user-friendly interface for real-time similarity computations and parameter tuning.

## Repository Structure

```
Plagiarism-Detection-Winnowing/
├── documents/
│   ├── my_paper.pdf                # Final research paper in IEEE format
│   ├── my_paper.tex                # LaTeX source of the paper
│   ├── presentation.pptx           # Final presentation slides
│   └── references.bib              # BibTeX references
├── src/
│   ├── main.py                     # Main script to compute similarities and predict plagiarism
│   ├── hashing.py                  # Rolling hash and winnowing implementation
│   ├── embeddings.py               # Code for generating semantic embeddings
│   ├── classifier.py               # XGBoost classifier training and prediction
│   ├── app.py                      # Streamlit application code
│   ├── preprocess.py               # Text preprocessing functions
│   ├── similarity.py               # Similarity computation functions
│   ├── winnowing.py                # Winnowing algorithm implementation
│   ├── graph.py                    # Graph plotting functions
│   └── utils.py                    # Utility functions
├── tests/
│   └── test_main.py                # Unit tests for main.py
├── figures/
│   ├── figure1.png                 # Architecture diagram
│   ├── lex_sem_plot.png            # Lexical vs. Semantic similarity plot
│   ├── winnowing_analogy.png       # Winnowing algorithm analogy diagram
│   └── streamlit_screenshot.png    # Screenshot of the Streamlit app interface
├── data/
│   ├── text1.txt                   # Sample text file 1
│   ├── text2.txt                   # Sample text file 2
│   ├── pairs.csv                    # Dataset pairs for training/testing
│   ├── text1_backup.txt            # Backup of text1.txt
│   └── text2_backup.txt            # Backup of text2.txt
├── Research\ Paper/                # Directory containing research paper documents
├── architecture.html               # HTML version of the architecture diagram
├── style.css                       # CSS styles for the Streamlit app
├── plagiarism_model.pkl            # Trained plagiarism detection model
├── scores_data.pkl                 # Serialized similarity scores data
├── README.md                        # This README file
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Git ignore file
```

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Git**

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Shashankk99/Plagiarism-Detection-Winnowing.git
   cd Plagiarism-Detection-Winnowing
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Running the Plagiarism Detection Script

The `main.py` script computes lexical and semantic similarities and predicts plagiarism.

```bash
python src/main.py --text1 "Sample text one." --text2 "Another text example."
```

**Parameters:**

- `--text1`: The first text input.
- `--text2`: The second text input.
- `--k`: (Optional) k-gram size for rolling hash (default: 5).
- `--w`: (Optional) Window size for winnowing (default: 4).

### 2. Training the XGBoost Classifier

Before running predictions, ensure the classifier is trained. This can be done by executing the `classifier.py` script.

```bash
python src/classifier.py
```

### 3. Launching the Streamlit Application

The Streamlit app provides an interactive interface for real-time similarity computations.

```bash
streamlit run src/app.py
```

Open the provided URL in your browser to access the application.

## Code Structure and Explanation

### `src/hashing.py`

Implements the rolling hash and winnowing algorithms to generate lexical fingerprints.

```python
def rolling_hash(text, k=5):
    """Compute rolling hashes for all k-grams in text."""
    hashes = []
    for i in range(len(text)-k+1):
        kgram = text[i:i+k]
        hashes.append(hash(kgram))
    return hashes

def winnowing(hashes, w=4):
    """Apply winnowing algorithm to select fingerprints."""
    fingerprints = set()
    min_hash = None
    min_pos = -1
    for i in range(len(hashes) - w + 1):
        window = hashes[i:i+w]
        current_min = min(window)
        pos = window.index(current_min) + i
        if current_min != min_hash:
            fingerprints.add(current_min)
            min_hash = current_min
            min_pos = pos
    return fingerprints
```

### `src/embeddings.py`

Generates semantic embeddings using Sentence-BERT.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_embeddings(text1, text2):
    """Generate embeddings for two texts."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2])
    return embeddings

def compute_semantic_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return sim * 100  # Convert to percentage
```

### `src/classifier.py`

Trains and saves the XGBoost classifier.

```python
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train_classifier(X, y):
    """Train XGBoost classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    joblib.dump(model, 'classifier.joblib')
    return model

if __name__ == "__main__":
    # Example dataset
    X = [[65.0, 68.0], [72.5, 63.3], [65.0, 59.0], [72.0, 55.0]]
    y = [0, 1, 0, 1]  # 1: Plagiarized, 0: Not Plagiarized
    train_classifier(X, y)
```

### `src/main.py`

Integrates hashing, embeddings, and classification to predict plagiarism.

```python
import argparse
from hashing import rolling_hash, winnowing
from embeddings import get_embeddings, compute_semantic_similarity
import joblib

def compute_lexical_similarity(text1, text2, k=5, w=4):
    hashes1 = rolling_hash(text1, k)
    fingerprints1 = winnowing(hashes1, w)
    hashes2 = rolling_hash(text2, k)
    fingerprints2 = winnowing(hashes2, w)
    overlap = fingerprints1.intersection(fingerprints2)
    similarity = (len(overlap) / max(len(fingerprints1), len(fingerprints2))) * 100
    return similarity

def main(text1, text2, k=5, w=4):
    lexical_sim = compute_lexical_similarity(text1, text2, k, w)
    emb1, emb2 = get_embeddings(text1, text2)
    semantic_sim = compute_semantic_similarity(emb1, emb2)
    
    model = joblib.load('classifier.joblib')
    prediction = model.predict([[lexical_sim, semantic_sim]])[0]
    
    print(f"Lexical Similarity: {lexical_sim:.2f}%")
    print(f"Semantic Similarity: {semantic_sim:.2f}%")
    print(f"Plagiarism Prediction: {'Plagiarized' if prediction == 1 else 'Not Plagiarized'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plagiarism Detection Script")
    parser.add_argument('--text1', type=str, required=True, help='First text input')
    parser.add_argument('--text2', type=str, required=True, help='Second text input')
    parser.add_argument('--k', type=int, default=5, help='k-gram size for rolling hash')
    parser.add_argument('--w', type=int, default=4, help='Window size for winnowing')
    args = parser.parse_args()
    main(args.text1, args.text2, args.k, args.w)
```

### `src/app.py`

Streamlit application for interactive plagiarism detection.

```python
import streamlit as st
from hashing import rolling_hash, winnowing
from embeddings import get_embeddings, compute_semantic_similarity
import joblib

def compute_lexical_similarity(text1, text2, k=5, w=4):
    hashes1 = rolling_hash(text1, k)
    fingerprints1 = winnowing(hashes1, w)
    hashes2 = rolling_hash(text2, k)
    fingerprints2 = winnowing(hashes2, w)
    overlap = fingerprints1.intersection(fingerprints2)
    similarity = (len(overlap) / max(len(fingerprints1), len(fingerprints2))) * 100
    return similarity

def main():
    st.title("Hybrid Lexical-Semantic Plagiarism Detection")
    
    text1 = st.text_area("Input Text 1")
    text2 = st.text_area("Input Text 2")
    
    k = st.slider("k-gram size", min_value=3, max_value=10, value=5)
    w = st.slider("Window size for winnowing", min_value=3, max_value=10, value=4)
    
    if st.button("Compute Similarity"):
        if text1 and text2:
            lexical_sim = compute_lexical_similarity(text1, text2, k, w)
            emb1, emb2 = get_embeddings(text1, text2)
            semantic_sim = compute_semantic_similarity(emb1, emb2)
            
            model = joblib.load('classifier.joblib')
            prediction = model.predict([[lexical_sim, semantic_sim]])[0]
            
            st.write(f"**Lexical Similarity:** {lexical_sim:.2f}%")
            st.write(f"**Semantic Similarity:** {semantic_sim:.2f}%")
            st.write(f"**Plagiarism Prediction:** {'Plagiarized' if prediction == 1 else 'Not Plagiarized'}")
        else:
            st.warning("Please enter both texts.")

if __name__ == "__main__":
    main()
```

### `src/utils.py`

Contains utility functions (if any). You can expand this as needed.

```python
# utils.py

def preprocess_text(text):
    """Normalize text by lowercasing and removing punctuation."""
    import string
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
```

### `tests/test_main.py`

Unit tests for `main.py`.

```python
import unittest
from src.main import compute_lexical_similarity
from src.hashing import rolling_hash, winnowing

class TestPlagiarismDetection(unittest.TestCase):
    def test_lexical_similarity(self):
        text1 = "This is a sample text for plagiarism detection."
        text2 = "This text is a sample for detecting plagiarism."
        similarity = compute_lexical_similarity(text1, text2, k=5, w=4)
        self.assertTrue(60 <= similarity <= 80)  # Adjust based on expected similarity

if __name__ == '__main__':
    unittest.main()
```

### `requirements.txt`

Ensure you have a `requirements.txt` file with all necessary dependencies.

```plaintext
xgboost
joblib
sentence-transformers
scikit-learn
streamlit
```

### `.gitignore`

To prevent accidentally committing unwanted files, ensure your `.gitignore` includes the following:

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
ENV/
env/
env.bak/
venv.bak/

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyderworkspace

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# Backup files
*_backup.txt

# Model and data artifacts
*.pkl
*.h5
# Uncomment if you want to ignore all CSV files
# *.csv

# Miscellaneous
.DS_Store
Thumbs.db
```

## Running Tests

To run the unit tests, navigate to the root directory and execute:

```bash
python -m unittest discover tests
```

## Demonstration: Streamlit Application

The Streamlit app allows real-time parameter tuning and immediate feedback.

![Streamlit App Interface](figures/streamlit_screenshot.png)

## Additional Notes

- **Figures Directory**: Ensure all image files (`figure1.png`, `lex_sem_plot.png`, `winnowing_analogy.png`, `streamlit_screenshot.png`) are placed in the `figures/` directory.
- **Research Paper**: The `Research Paper/` directory contains all necessary documents related to your research.
- **Backup Files**: Backup files like `text1_backup.txt` and `text2_backup.txt` are excluded from version control via `.gitignore`.

## Acknowledgments

The author thanks colleagues at Stevens Institute of Technology, the open-source NLP community, and acknowledges any internal research support.

---
