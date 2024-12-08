# src/main.py

from preprocess import preprocess_text
from hashing import rolling_hash
from winnowing import winnow_hashes
from similarity import compute_similarity

def main():
    # Load and preprocess text files
    with open('../data/text1.txt', 'r') as file:
        text1 = file.read()
    with open('../data/text2.txt', 'r') as file:
        text2 = file.read()

    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)

    # Define k-gram size and window size
    k = 5
    window_size = 4

    # Generate hashes
    hashes1 = [rolling_hash(processed_text1[i:i+k]) for i in range(len(processed_text1) - k + 1)]
    hashes2 = [rolling_hash(processed_text2[i:i+k]) for i in range(len(processed_text2) - k + 1)]

    # Extract fingerprints
    fingerprints1 = winnow_hashes(hashes1, window_size)
    fingerprints2 = winnow_hashes(hashes2, window_size)

    # Compute similarity score
    similarity_score = compute_similarity(fingerprints1, fingerprints2)
    print(f"Similarity Score: {similarity_score}")

if __name__ == "__main__":
    main()
