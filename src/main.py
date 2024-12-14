# src/main.py

import sys
import os
from .preprocess import preprocess_text
from .hashing import rolling_hash
from .winnowing import winnow_hashes
from .similarity import compute_similarity

def main():
    try:
        # Define data directory relative to the script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

        # Paths to the text files
        file1_path = os.path.join(data_dir, 'text1.txt')
        file2_path = os.path.join(data_dir, 'text2.txt')

        # Load and preprocess text files
        with open(file1_path, 'r') as file:
            text1 = file.read()
        with open(file2_path, 'r') as file:
            text2 = file.read()

        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)

        if not processed_text1 or not processed_text2:
            print("One or both input files are empty after preprocessing.")
            sys.exit(1)

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
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error during hashing: {e}")
        sys.exit(1)
    except AttributeError as e:
        print(f"Type error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
