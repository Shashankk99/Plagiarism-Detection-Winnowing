# src/hashing.py

def rolling_hash(text):
    """
    Compute a hash for a given k-gram string.

    Complexity:
    - Computing a hash for a single k-gram is O(k).
    - If we're doing this for all k-grams in a text of length n,
      total complexity is O(n) *assuming k is fixed and much smaller than n.*
    """
    if not text:
        raise ValueError("Empty k-gram is not allowed.")
    if not isinstance(text, str):
        raise AttributeError("k-gram must be a string.")
    return hash(text)

if __name__ == "__main__":
    sample_kgram = "hello"
    print(f"Hash for '{sample_kgram}': {rolling_hash(sample_kgram)}")
