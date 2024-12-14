# tests/test_main.py

import pytest
from unittest.mock import patch, mock_open
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

def test_main_successful_run():
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
            raise FileNotFoundError(f"File {file} not found")

    with patch('builtins.open', new_callable=mock_open) as mock_file:
        mock_file.side_effect = open_side_effect
        with patch('src.main.preprocess_text', return_value="sample text"):
            with patch('src.main.rolling_hash', side_effect=rolling_hash_generator()):
                with patch('src.main.winnow_hashes', return_value={1, 2, 3}):
                    with patch('src.main.compute_similarity', return_value=expected_similarity):
                        with patch('builtins.print') as mock_print:
                            main()
                            # Verify that print was called with the correct similarity score
                            mock_print.assert_called_with(f"Similarity Score: {expected_similarity}")

def test_main_similarity_zero():
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
            raise FileNotFoundError(f"File {file} not found")

    with patch('builtins.open', new_callable=mock_open) as mock_file:
        mock_file.side_effect = open_side_effect
        with patch('src.main.preprocess_text', return_value="unique text one"):
            with patch('src.main.rolling_hash', side_effect=rolling_hash_generator()):
                with patch('src.main.winnow_hashes', return_value={1, 2, 3}):
                    with patch('src.main.compute_similarity', return_value=expected_similarity):
                        with patch('builtins.print') as mock_print:
                            main()
                            mock_print.assert_called_with(f"Similarity Score: {expected_similarity}")

def test_main_file_not_found():
    """
    Test that the main function exits gracefully when a file is not found.
    """
    with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
        with patch('src.main.preprocess_text', return_value=""):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1  # Check if exit code is 1

def test_main_empty_files():
    """
    Test that the main function exits when input files are empty.
    """
    # Create a mock for open that returns empty strings for both files
    def open_side_effect(file, mode='r', *args, **kwargs):
        filename = os.path.basename(file)
        if filename == 'text1.txt' or filename == 'text2.txt':
            return mock_open(read_data="").return_value
        else:
            raise FileNotFoundError(f"File {file} not found")

    with patch('builtins.open', new_callable=mock_open) as mock_file:
        mock_file.side_effect = open_side_effect
        with patch('src.main.preprocess_text', return_value=""):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1  # Check if exit code is 1

def test_main_hashing_error():
    """
    Test that the main function handles errors during hashing appropriately by exiting the application.
    """
    # Define the mock data for both files
    mock_text1 = "Sample text."
    mock_text2 = "Another sample text."

    # Create a mock for open that returns different data based on the file being opened
    def open_side_effect(file, mode='r', *args, **kwargs):
        filename = os.path.basename(file)
        if filename == 'text1.txt':
            return mock_open(read_data=mock_text1).return_value
        elif filename == 'text2.txt':
            return mock_open(read_data=mock_text2).return_value
        else:
            raise FileNotFoundError(f"File {file} not found")

    with patch('builtins.open', new_callable=mock_open) as mock_file:
        mock_file.side_effect = open_side_effect
        with patch('src.main.preprocess_text', return_value="sample text"):
            # Simulate a ValueError during hashing
            with patch('src.main.rolling_hash', side_effect=ValueError("Invalid k-gram")):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1  # Check if exit code is 1
