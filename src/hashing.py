def rolling_hash(text):
    """
    Compute a simple hash for a k-gram (substring).
    Raises:
        ValueError: If the input text is empty.
        AttributeError: If the input text is not a string.
    """
    if not text:
        raise ValueError("Empty k-gram is not allowed.")
    if not isinstance(text, str):
        raise AttributeError("k-gram must be a string.")
    return hash(text)

if __name__ == "__main__":
    # Example usage
    sample_kgram = "hello"
    print(f"Hash for '{sample_kgram}': {rolling_hash(sample_kgram)}")
