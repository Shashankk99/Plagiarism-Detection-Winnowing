# src/similarity.py

from sentence_transformers import SentenceTransformer, util

def compute_similarity(fingerprints1, fingerprints2):
    common = fingerprints1.intersection(fingerprints2)
    total = max(len(fingerprints1), len(fingerprints2))
    similarity = (len(common) / total) * 100 if total > 0 else 0
    return similarity

def compute_semantic_similarity(text1, text2):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
    semantic_similarity = float(cosine_scores.item()) * 100
    return semantic_similarity
