# src/similarity.py

from sentence_transformers import SentenceTransformer, util

def compute_similarity(fingerprints1, fingerprints2):
    """
    Compute lexical similarity as:
    (# of common fingerprints) / (max(len(fp1), len(fp2))) * 100%

    Complexity:
    - Intersection of sets is O(min(len(fp1), len(fp2))) operation.
    - Overall O(n) where n is the size of the fingerprint sets.
    """
    common = fingerprints1.intersection(fingerprints2)
    total = max(len(fingerprints1), len(fingerprints2))
    similarity = (len(common) / total) * 100 if total > 0 else 0
    return similarity

def compute_semantic_similarity(text1, text2):
    """
    Compute semantic similarity using a sentence transformer model.

    Complexity:
    - Encoding two texts is O(n) where n is proportional to text length, due to model inference.
    - Cosine similarity computation is O(1) after embeddings.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
    semantic_similarity = float(cosine_scores.item()) * 100
    return semantic_similarity
