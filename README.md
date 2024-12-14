# **Plagiarism Detection Using Winnowing Algorithm**

## **Overview**
Welcome to my plagiarism detection project! The idea here is simple yet powerful: to detect textual similarities between documents and identify potential plagiarism using the **Winnowing Algorithm**. This project processes text files, generates hashed fingerprints, and computes similarity scores between documents. It’s a step towards automating what can otherwise be a tedious manual process.

## **What It Does**
This system takes two text files as input and processes them in multiple stages:
1. **Text Preprocessing**: Cleans the text by:
   - Converting everything to lowercase.
   - Removing punctuation.
   - Eliminating stopwords (common words like "the," "is," etc.).
2. **Hashing**: Uses a rolling hash function to create unique numeric representations of substrings (k-grams).
3. **Winnowing**: Identifies unique fingerprints from the hashed substrings using a sliding window technique.
4. **Similarity Calculation**: Compares fingerprints from two documents and computes a similarity score. A score of `1.0` means the two texts are identical, while lower scores indicate less similarity.

## **So Far, What We Have Implemented**
Here’s a summary of what’s up and running:
- **Preprocessing**: The text normalization, tokenization, and stopword removal are implemented and working perfectly.
- **Hashing**: Rolling hash function is successfully generating hashes for k-grams.
- **Winnowing Algorithm**: Extracts unique fingerprints from the hash list using a sliding window.
- **Similarity Computation**: Calculates a meaningful similarity score between two files, showing how similar or different they are.
- **A Fully Functional Pipeline**: Everything is integrated! You can input your text files, and the system will walk you through preprocessing to similarity scoring.

## **How to Run It**
1. Clone the repository.
2. Make sure Python and `nltk` are installed.
3. Create your text files in the `data/` folder (e.g., `text1.txt` and `text2.txt`).
4. Run the `main.py` file:
   ```bash
   python main.py
   ```

## **Current Status**
The core functionality is complete! The system:
- Processes text data.
- Computes similarity scores accurately.
- Is well-structured and modular for easy updates or additional features.

## **What’s Next?**
- Writing the research paper.
- Testing with larger datasets and fine-tuning parameters like k-gram size and window size.
- Exploring additional algorithms (like Rabin-Karp and Suffix Trees) for comparison.

  ## Testing

This project includes a comprehensive suite of unit tests to ensure the reliability and correctness of each module. All test files are located in the `tests/` directory.

### Test Files

- `test_preprocess.py`: Tests for the text preprocessing functions.
- `test_hashing.py`: Tests for the hashing functions.
- `test_winnowing.py`: Tests for the Winnowing algorithm implementation.
- `test_similarity.py`: Tests for the similarity computation functions.
- `test_main.py`: Tests for the main application workflow.

### Running Tests

To execute all tests, ensure you're in the project root directory and run:

```bash
pytest tests/


---
