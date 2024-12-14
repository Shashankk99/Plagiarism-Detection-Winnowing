# src/winnowing.py
from .hashing import rolling_hash

def winnow_hashes(hashes, window_size):
    """
    Extract fingerprints from a list of hashes using the winnowing algorithm.
    Args:
        hashes (list): List of hash values.
        window_size (int): Size of the sliding window.
    Returns:
        set: Set of fingerprints.
    """
    fingerprints = set()
    for i in range(len(hashes) - window_size + 1):
        window = hashes[i:i + window_size]
        min_hash = min(window)
        fingerprints.add(min_hash)
    return fingerprints

if __name__ == "__main__":
    # Example usage
    sample_hashes = [83, 77, 96, 84, 22, 67, 98]
    window_size = 4
    fingerprints = winnow_hashes(sample_hashes, window_size)
    print(f"Fingerprints: {fingerprints}")
