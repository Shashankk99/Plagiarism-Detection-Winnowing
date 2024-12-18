# src/winnowing.py
from src.hashing import rolling_hash

def winnow_hashes(hashes, window_size):
    """
    Implement the winnowing algorithm:
    - Split the sequence of hashes into overlapping windows of size w.
    - Pick the minimum hash in each window as a fingerprint.

    Complexity:
    - For a list of n hashes, we slide a window of size w across it.
    - For each window, finding a minimum is O(w). Naive complexity is O(n*w).
    - Typically, w is small and considered constant relative to n, so O(n).

    Returns a set of fingerprints (unique minimal hashes).
    """
    fingerprints = set()
    for i in range(len(hashes) - window_size + 1):
        window = hashes[i:i + window_size]
        min_hash = min(window)
        fingerprints.add(min_hash)
    return fingerprints

if __name__ == "__main__":
    sample_hashes = [83, 77, 96, 84, 22, 67, 98]
    window_size = 4
    fingerprints = winnow_hashes(sample_hashes, window_size)
    print(f"Fingerprints: {fingerprints}")
