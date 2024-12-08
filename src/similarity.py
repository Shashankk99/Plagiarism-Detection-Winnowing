# src/similarity.py

def compute_similarity(fingerprints1, fingerprints2):
    """
    Compute the Jaccard similarity between two sets of fingerprints.
    Args:
        fingerprints1 (set): Set of fingerprints from the first document.
        fingerprints2 (set): Set of fingerprints from the second document.
    Returns:
        float: Jaccard similarity score.
    """
    intersection = fingerprints1.intersection(fingerprints2)
    union = fingerprints1.union(fingerprints2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

if __name__ == "__main__":
    # Example usage
    fingerprints1 = {77, 22, 96, 84}
    fingerprints2 = {22, 67, 98, 77}
    similarity = compute_similarity(fingerprints1, fingerprints2)
    print(f"Similarity Score: {similarity}")
