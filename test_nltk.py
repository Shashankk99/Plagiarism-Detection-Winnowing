# test_nltk.py

import sys
import os

# Determine the absolute path to the 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")

# Add 'src' to sys.path if it's not already included
if src_dir not in sys.path:
    sys.path.append(src_dir)

from preprocess import preprocess_text

# Now you can proceed with testing
sample_text = "The children are playing with their toys."
processed_text = preprocess_text(sample_text)
print(f"Original Text: {sample_text}")
print(f"Processed Text: {processed_text}")
