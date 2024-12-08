# src/hashing.py

def rolling_hash(kgram):
    """
    Compute a simple hash for a k-gram (substring).
    """
    return hash(kgram)

if __name__ == "__main__":
    # Example usage
    sample_kgram = "hello"
    print(f"Hash for '{sample_kgram}': {rolling_hash(sample_kgram)}")
