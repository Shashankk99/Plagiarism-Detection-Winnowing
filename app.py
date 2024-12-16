# app.py

import streamlit as st
import pickle
from src.preprocess import preprocess_text
from src.similarity import compute_lexical_similarity, compute_semantic_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model with caching to improve performance
@st.cache_resource
def load_model():
    try:
        with open('plagiarism_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

clf = load_model()

st.title("Plagiarism Detection with Winnowing")

# User inputs
text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")

# Parameters with default values
k = st.slider("k-gram size (k)", min_value=3, max_value=10, value=7)
window_size = st.slider("Winnowing window size", min_value=3, max_value=10, value=5)

if st.button("Compute Similarity & Predict"):
    if not text1.strip() or not text2.strip():
        st.error("Both texts are required!")
    else:
        try:
            # Preprocess texts
            processed_text1 = preprocess_text(text1)
            processed_text2 = preprocess_text(text2)
            
            if not processed_text1 or not processed_text2:
                st.error("Preprocessing resulted in empty tokens.")
            else:
                # Compute similarity scores
                lexical_similarity = compute_lexical_similarity(processed_text1, processed_text2, k, window_size)
                semantic_similarity = compute_semantic_similarity(text1, text2)
                
                # Create feature vector
                feature_vector = [[lexical_similarity, semantic_similarity]]
                
                # Make prediction
                if clf:
                    prediction = clf.predict(feature_vector)[0]
                    confidence = clf.predict_proba(feature_vector)[0][prediction]
                    
                    # Display results
                    st.write(f"**Lexical Similarity:** {lexical_similarity:.2f}%")
                    st.write(f"**Semantic Similarity:** {semantic_similarity:.2f}%")
                    if prediction == 1:
                        st.write(f"**Prediction:** Plagiarized (**Confidence:** {confidence * 100:.2f}%)")
                    else:
                        st.write(f"**Prediction:** Not Plagiarized (**Confidence:** {confidence * 100:.2f}%)")
                else:
                    st.error("Model is not loaded correctly.")
        except Exception as e:
            logging.error(f"Error during processing: {e}")
            st.error(f"An error occurred during processing: {e}")
