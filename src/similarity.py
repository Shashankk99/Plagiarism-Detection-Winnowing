# src/similarity.py
def compute_similarity(fingerprints1, fingerprints2):
    """
    Compute the similarity score based on the fingerprints.
    Args:
        fingerprints1 (set): Fingerprints from the first text.
        fingerprints2 (set): Fingerprints from the second text.
    Returns:
        float: Similarity score as a percentage.
    """
    common = fingerprints1.intersection(fingerprints2)
    total = max(len(fingerprints1), len(fingerprints2))
    similarity = (len(common) / total) * 100 if total > 0 else 0
    return similarity
