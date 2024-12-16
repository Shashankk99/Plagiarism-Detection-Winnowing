# src/hashing.py

def rolling_hash(text):
    if not text:
        raise ValueError("Empty k-gram is not allowed.")
    if not isinstance(text, str):
        raise AttributeError("k-gram must be a string.")
    return hash(text)

if __name__ == "__main__":
    sample_kgram = "hello"
    print(f"Hash for '{sample_kgram}': {rolling_hash(sample_kgram)}")
