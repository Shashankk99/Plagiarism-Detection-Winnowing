# Hybrid Lexical-Semantic Plagiarism Detection

## Overview

Ensuring the originality of academic and professional content is crucial. Traditional plagiarism detection methods, which rely heavily on **lexical overlap**, often fail when confronted with paraphrased passages that maintain meaning but alter wording. This project presents a **hybrid approach** that integrates **lexical fingerprinting** (using rolling hash and winnowing) with **semantic embeddings** obtained from transformer-based models. By combining both lexical and semantic similarity scores, our system more effectively identifies plagiarism across a spectrum of scenarios, from direct copying to subtle conceptual paraphrasing.

**Key Advancements:**
- **Hybrid Detection:** Unites surface-level lexical matches with deeper semantic understanding.
- **Multiple Algorithm Comparison:** Evaluates Logistic Regression, Random Forest, and XGBoost for the classification stage, ensuring robust and well-founded algorithm selection.
- **Complexity Analysis:** Provides insight into the computational efficiency of hashing, winnowing, and semantic embedding generation, confirming scalability.
- **User-Friendly Interface:** Offers an interactive Streamlit app for real-time experimentation and parameter tuning.

## Features

- **Lexical Fingerprinting:**  
  Implemented via rolling hash and winnowing to produce stable fingerprints, capturing direct textual overlaps.

- **Semantic Embeddings:**  
  Utilizes transformer-based sentence encoders (e.g., Sentence-BERT) to detect conceptual and contextual similarities.

- **Hybrid Classification:**  
  Combines lexical and semantic similarity scores. After evaluating multiple algorithms, we leverage XGBoost for robust, data-driven classification of plagiarized vs. non-plagiarized text.

- **Real-Time Interaction:**  
  The Streamlit application enables immediate feedback on parameter changes (like k-gram size or window size) and visualizes both lexical and semantic similarity outputs.

- **Comprehensive Testing:**  
  A suite of unit tests ensures that preprocessing, hashing, semantic scoring, and classification components work reliably.

## Repository Structure

```plaintext
Plagiarism-Detection-Winnowing/
├── documents/
│   ├── my_paper.pdf                # Final research paper (IEEE format)
│   ├── my_paper.tex                # LaTeX source of the paper
│   ├── presentation.pptx           # Final presentation slides
│   └── references.bib              # BibTeX references
├── src/
│   ├── main.py                     # Main script for computing similarities & predicting plagiarism
│   ├── hashing.py                  # Rolling hash & winnowing implementations
│   ├── embeddings.py               # Semantic embedding generation
│   ├── classifier.py               # Training & saving the XGBoost (and other) classifiers
│   ├── app.py                      # Streamlit app for interactive parameter tuning
│   ├── preprocess.py               # Text preprocessing functions
│   ├── similarity.py               # Similarity computations (lexical & semantic)
│   ├── winnowing.py                # Alternative winnowing implementations if needed
│   ├── graph.py                    # Utilities for plotting data distributions
│   └── utils.py                    # General utility functions
├── tests/
│   └── test_main.py                # Unit tests for main logic
├── figures/
│   ├── figure1.png                 # System architecture diagram
│   ├── lex_sem_plot.png            # Lexical vs. semantic similarity distribution
│   ├── winnowing_analogy.png       # Visual analogy for the winnowing algorithm
│   └── streamlit_screenshot.png    # Screenshot of the Streamlit interface
├── data/
│   ├── text1.txt                   # Sample text 1
│   ├── text2.txt                   # Sample text 2
│   ├── pairs.csv                   # Dataset pairs (e.g., Quora Question Pairs subset)
│   ├── text1_backup.txt            # Backup of text1.txt
│   └── text2_backup.txt            # Backup of text2.txt
├── Research\ Paper/                # Directory for research paper drafts & related docs
├── architecture.html               # HTML rendition of the system architecture
├── style.css                       # CSS styling for Streamlit app
├── plagiarism_model.pkl            # Trained plagiarism detection model
├── scores_data.pkl                 # Precomputed scores for plotting or analysis
├── README.md                       # Project README
├── requirements.txt                # Dependencies list
└── .gitignore                      # Git ignore file
```

## Installation

### Prerequisites
- **Python 3.7+**
- **Git**

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Shashankk99/Plagiarism-Detection-Winnowing.git
   cd Plagiarism-Detection-Winnowing
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Classifier
Before performing predictions, ensure the model is trained:
```bash
python src/classifier.py
```
This trains the XGBoost classifier using sample data and saves `plagiarism_model.pkl`.

### 2. Running the Main Script
Compute lexical & semantic similarities and predict plagiarism:
```bash
python src/main.py --text1 "Sample text one." --text2 "Another text example."
```
**Optional Arguments:**
- `--k`: k-gram size for rolling hash (default: 5)
- `--w`: Window size for winnowing (default: 4)

### 3. Launching the Streamlit App
Interactively tune parameters and view results in real-time:
```bash
streamlit run src/app.py
```
Open the displayed URL in your browser to access the app.

### 4. Running Unit Tests
Verify that all components work as intended:
```bash
python -m unittest discover tests
```

## Complexity and Performance
- **Hashing & Winnowing:** O(n) for a text of length n  
- **Embedding Generation:** O(n) due to model inference  
- **Classification Inference:** O(1) per text pair  
The approach scales efficiently to longer texts without significant performance loss.

## Visual Demonstrations
- **Architecture Diagram:** Displays how lexical and semantic analyses integrate before classification (`figure1.png`).
- **Lexical vs. Semantic Plot:** Shows distribution of plagiarized vs. non-plagiarized pairs (`lex_sem_plot.png`).
- **Streamlit App Screenshot:** Provides a glimpse of the interactive interface (`streamlit_screenshot.png`).

## Future Work
- Evaluating on specialized plagiarism datasets for more domain-specific benchmarks.
- Experimenting with larger or multilingual transformer models.
- Fine-tuning hyperparameters and exploring additional classifiers for incremental performance gains.

## Acknowledgments
We acknowledge contributions from the Stevens Institute of Technology community, the open-source NLP frameworks, and datasets that made this project possible.

## License
This project is available under the [MIT License](LICENSE).

---

### Notes:
- **Code Blocks:** Ensure that all code snippets and command-line instructions are enclosed within appropriate code blocks using triple backticks (```) with the relevant language identifier (e.g., `bash`, `python`, `plaintext`).
  
- **Repository Structure:** The repository structure is displayed using a `plaintext` code block for better readability.

- **Links:** Make sure that the `[MIT License](LICENSE)` link correctly points to your license file in the repository.
