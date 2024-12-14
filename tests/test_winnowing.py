from src.winnowing import winnow_hashes

def test_winnow_hashes_normal_case():
    """
    Test the winnow_hashes function with a standard set of hashes and a window size.
    """
    hashes = [77, 22, 33, 44, 77, 22]
    window_size = 3
    expected_fingerprints = {22, 33}  # Correct based on sliding windows
    assert winnow_hashes(hashes, window_size) == expected_fingerprints

def test_winnow_hashes_single_window():
    """
    Test the winnow_hashes function with a single window.
    """
    hashes = [10, 20, 30]
    window_size = 3
    expected_fingerprints = {10}
    assert winnow_hashes(hashes, window_size) == expected_fingerprints

def test_winnow_hashes_window_size_one():
    """
    Test the winnow_hashes function with window size of one.
    """
    hashes = [5, 3, 9, 1]
    window_size = 1
    expected_fingerprints = {5, 3, 9, 1}
    assert winnow_hashes(hashes, window_size) == expected_fingerprints

def test_winnow_hashes_empty_hashes():
    """
    Test the winnow_hashes function with an empty list of hashes.
    """
    hashes = []
    window_size = 3
    expected_fingerprints = set()
    assert winnow_hashes(hashes, window_size) == expected_fingerprints

def test_winnow_hashes_window_size_larger_than_hashes():
    """
    Test the winnow_hashes function when window size is larger than the list of hashes.
    """
    hashes = [10, 20]
    window_size = 3
    expected_fingerprints = set()
    assert winnow_hashes(hashes, window_size) == expected_fingerprints
