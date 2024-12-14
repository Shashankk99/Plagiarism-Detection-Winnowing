from src.similarity import compute_similarity

def test_full_overlap():
    """
    Test that the similarity score is 100.0 when both fingerprint sets are identical.
    """
    fingerprints1 = {1, 2, 3}
    fingerprints2 = {1, 2, 3}
    assert compute_similarity(fingerprints1, fingerprints2) == 100.0

def test_partial_overlap():
    """
    Test that the similarity score is correctly calculated when there is partial overlap.
    """
    fingerprints1 = {1, 2, 3}
    fingerprints2 = {3, 4, 5}
    expected_similarity = 33.33333333333333  # (1/3) * 100
    assert compute_similarity(fingerprints1, fingerprints2) == expected_similarity

def test_no_overlap():
    """
    Test that the similarity score is 0.0 when there is no overlap between fingerprints.
    """
    fingerprints1 = {1, 2, 3}
    fingerprints2 = {4, 5, 6}
    expected_similarity = 0.0
    assert compute_similarity(fingerprints1, fingerprints2) == expected_similarity
