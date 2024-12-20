import streamlit as st
import sys
import os
import pickle
import nltk

# Set NLTK data path explicitly
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Ensure required NLTK resources are downloaded
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from preprocess import preprocess_text
from hashing import rolling_hash
from winnowing import winnow_hashes
from similarity import compute_similarity, compute_semantic_similarity

# Load the trained model
model_path = os.path.join(current_dir, "plagiarism_model.pkl")
with open(model_path, "rb") as f:
    clf = pickle.load(f)

st.title("Advanced Plagiarism Detection App")

st.write("""
This application uses a hybrid approach to detect plagiarism:
- **Lexical Similarity:** via rolling hash + winnowing.
- **Semantic Similarity:** via a transformer-based embedding model.
- **Classification:** Uses a model trained on the Quora Question Pairs dataset.
""")

input_method = st.radio("Choose input method:", ["Direct Text Input", "Upload Text Files"])

if input_method == "Direct Text Input":
    text1 = st.text_area("Enter Text 1:", height=200, value="This is a sample text.")
    text2 = st.text_area("Enter Text 2:", height=200, value="This is a sample text.")
else:
    uploaded_file1 = st.file_uploader("Upload first text file", type=["txt"])
    uploaded_file2 = st.file_uploader("Upload second text file", type=["txt"])
    text1, text2 = "", ""
    if uploaded_file1:
        text1 = uploaded_file1.read().decode("utf-8", errors="replace")
    if uploaded_file2:
        text2 = uploaded_file2.read().decode("utf-8", errors="replace")

st.write("### Adjust Parameters:")
k = st.slider("k (k-gram size)", min_value=3, max_value=10, value=7)
window_size = st.slider("window_size (for winnowing)", min_value=2, max_value=10, value=5)

if st.button("Compute Similarity & Predict"):
    if not text1.strip() or not text2.strip():
        st.error("Both texts are required!")
    else:
        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)

        # Compute lexical similarity
        if len(processed_text1) < k or len(processed_text2) < k:
            lexical_similarity_score = 0.0
        else:
            hashes1 = [rolling_hash(processed_text1[i:i+k]) for i in range(len(processed_text1) - k + 1)]
            hashes2 = [rolling_hash(processed_text2[i:i+k]) for i in range(len(processed_text2) - k + 1)]
            fingerprints1 = winnow_hashes(hashes1, window_size)
            fingerprints2 = winnow_hashes(hashes2, window_size)
            lexical_similarity_score = compute_similarity(fingerprints1, fingerprints2)

        # Compute semantic similarity
        semantic_similarity_score = compute_semantic_similarity(processed_text1, processed_text2)

        st.write("### Similarity Scores")
        st.write(f"**Lexical Similarity Score:** {lexical_similarity_score:.2f}%")
        st.write(f"**Semantic Similarity Score:** {semantic_similarity_score:.2f}%")

        # Make prediction
        features = [[lexical_similarity_score, semantic_similarity_score]]
        prediction = clf.predict(features)[0]
        prediction_proba = clf.predict_proba(features)[0][prediction]

        st.write("### Final Classification")
        if prediction == 1:
            st.success(f"**Predicted: Plagiarized** (Confidence: {prediction_proba*100:.2f}%)")
        else:
            st.info(f"**Predicted: Not Plagiarized** (Confidence: {prediction_proba*100:.2f}%)")
