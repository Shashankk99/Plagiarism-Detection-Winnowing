# tests/test_main.py

import pytest
from unittest.mock import patch, mock_open, Mock
from src.main import main
import os

def rolling_hash_generator():
    """
    A generator to provide rolling_hash outputs without exhausting.
    """
    i = 1
    while True:
        yield i
        i += 1

@pytest.fixture
def mock_sentence_transformer():
    """
    A fixture that mocks the SentenceTransformer class so no model files are actually loaded.
    We patch it in src.similarity since that's where SentenceTransformer is imported and used.
    """
    with patch('src.similarity.SentenceTransformer') as mock_model_cls:
        mock_model_instance = Mock()
        # Mock the encode method to return dummy embeddings
        mock_model_instance.encode.return_value = [[0.1]*384, [0.1]*384]
        mock_model_cls.return_value = mock_model_instance
        yield mock_model_cls

def test_main_successful_run(mock_sentence_transformer):
    """
    Simulate a successful execution of the main function and verify that the correct similarity score is printed.
    """
    expected_similarity = 100.0  # Since both texts are identical

    # Define the mock data for both files
    mock_text1 = "This is a simple text."
    mock_text2 = "This is a simple text."

    # Create a mock for open that returns different data based on the file being opened
    def open_side_effect(file, mode='r', *args, **kwargs):
        filename = os.path.basename(file)
        if filename == 'text1.txt':
            return mock_open(read_data=mock_text1).return_value
        elif filename == 'text2.txt':
            return mock_open(read_data=mock_text2).return_value
        else:
            # For any other file, return the real open to allow model loading if needed
            return open(file, mode, *args, **kwargs)

    with patch('builtins.open', new_callable=mock_open) as mock_file:
        mock_file.side_effect = open_side_effect
        with patch('src.main.preprocess_text', return_value="sample text"):
            with patch('src.main.rolling_hash', side_effect=rolling_hash_generator()):
                with patch('src.main.winnow_hashes', return_value={1, 2, 3}):
                    with patch('src.main.compute_similarity', return_value=expected_similarity):
                        with patch('builtins.print') as mock_print:
                            main()
                            # Verify that print was called with the correct similarity score
                            mock_print.assert_any_call("Lexical Similarity Score: 100.00%")
                            mock_print.assert_any_call("Semantic Similarity Score: 100.00%")

def test_main_similarity_zero(mock_sentence_transformer):
    """
    Test that the main function correctly computes a similarity score of 0.0 when there is no overlap between fingerprints.
    """
    expected_similarity = 0.0  # As percentage

    # Define the mock data for both files
    mock_text1 = "Unique text one."
    mock_text2 = "Completely different text."

    # Create a mock for open that returns different data based on the file being opened
    def open_side_effect(file, mode='r', *args, **kwargs):
        filename = os.path.basename(file)
        if filename == 'text1.txt':
            return mock_open(read_data=mock_text1).return_value
        elif filename == 'text2.txt':
            return mock_open(read_data=mock_text2).return_value
        else:
            return open(file, mode, *args, **kwargs)

    with patch('builtins.open', new_callable=mock_open) as mock_file:
        mock_file.side_effect = open_side_effect
        with patch('src.main.preprocess_text', return_value="unique text one"):
            with patch('src.main.rolling_hash', side_effect=rolling_hash_generator()):
                with patch('src.main.winnow_hashes', return_value={1, 2, 3}):
                    with patch('src.main.compute_similarity', return_value=expected_similarity):
                        with patch('builtins.print') as mock_print:
                            main()
                            mock_print.assert_any_call("Lexical Similarity Score: 0.00%")
                            mock_print.assert_any_call("Semantic Similarity Score: 100.00%")
                            # Note: Semantic similarity remains 100% because we're mocking identical embeddings.
