import pytest
from src.hashing import rolling_hash

def test_rolling_hash_empty_kgram():
    """
    Test that the rolling_hash function raises a ValueError when given an empty k-gram.
    """
    with pytest.raises(ValueError):
        rolling_hash("")

def test_rolling_hash_type_error():
    """
    Test that the rolling_hash function raises an AttributeError when a non-string type is passed.
    """
    with pytest.raises(AttributeError):
        rolling_hash(12345)
