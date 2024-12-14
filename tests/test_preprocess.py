# tests/test_preprocess.py

import pytest
from src.preprocess import preprocess_text

def test_preprocess_text():
    """
    Test the preprocess_text function with sample input.
    """
    input_text = "This is a SAMPLE Text.\nNew line here."
    expected_output = "sample text new line"
    assert preprocess_text(input_text) == expected_output

def test_preprocess_text_empty_input():
    """
    Test that preprocess_text returns an empty string when given empty input.
    """
    input_text = ""
    expected_output = ""
    assert preprocess_text(input_text) == expected_output

def test_preprocess_text_only_stopwords():
    """
    Test that preprocess_text returns an empty string when input contains only stopwords.
    """
    input_text = "This is a the and or."
    expected_output = ""
    assert preprocess_text(input_text) == expected_output
