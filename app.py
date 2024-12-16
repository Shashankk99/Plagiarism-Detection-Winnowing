import streamlit as st
import sys
import os
import pickle

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
- **Classification:** Uses an XGBoost model trained on the Quora Question Pairs dataset to predict whether the pair of texts is considered "plagiarized" (paraphrased/duplicate) or not.

You can experiment with parameters and see how they affect the similarity scores and final prediction.
""")

# User can choose input method
input_method = st.radio("Choose input method:", ["Direct Text Input", "Upload Text Files"])

if input_method == "Direct Text Input":
    text1 = st.text_area("Enter Text 1:", height=200, value="This is a sample text.")
    text2 = st.text_area("Enter Text 2:", height=200, value="This is a sample text.")
else:
    uploaded_file1 = st.file_uploader("Upload first text file", type=["txt"])
    uploaded_file2 = st.file_uploader("Upload second text file", type=["txt"])
    text1 = ""
    text2 = ""
    if uploaded_file1 is not None:
        text1 = uploaded_file1.read().decode("utf-8", errors="replace")
    if uploaded_file2 is not None:
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

        if not processed_text1 or not processed_text2:
            st.error("One or both texts are empty after preprocessing. Please provide more substantial text.")
        else:
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

            # Make a prediction using the classifier
            features = [[lexical_similarity_score, semantic_similarity_score]]
            prediction = clf.predict(features)[0]
            prediction_proba = clf.predict_proba(features)[0][prediction]

            st.write("### Final Classification")
            if prediction == 1:
                st.success(f"**Predicted: Plagiarized** (Confidence: {prediction_proba*100:.2f}%)")
            else:
                st.info(f"**Predicted: Not Plagiarized** (Confidence: {prediction_proba*100:.2f}%)")

            st.write("""
            **Note:** This prediction is based on the trained XGBoost model using the Quora Question Pairs dataset as a proxy for paraphrased/plagiarized content.
            Adjusting k and window_size or providing different texts will affect the outcome.
            """)
