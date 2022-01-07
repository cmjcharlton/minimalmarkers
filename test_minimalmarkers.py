

from minimalmarkers import load_patterns, find_best_patterns, \
                           calculate_best_possible_score, Patterns


def test_minimal_markers():
    """Validates that the AppleGenotypes.csv file is loaded
       and processed correctly
    """
    patterns = load_patterns("example/AppleGenotypes.csv")

    assert patterns is not None
    assert type(patterns) == Patterns

    n_varieties = 260
    n_patterns = 1269

    assert len(patterns.varieties) == n_varieties
    assert len(patterns.ids) == n_patterns
    assert len(patterns.mafs) == n_patterns
    assert patterns.patterns.shape == (n_patterns, n_varieties)

    best_score = calculate_best_possible_score(patterns)

    assert best_score == 33669

    best_patterns = find_best_patterns(patterns)

    # The score from the last pattern must give the best
    # possible score
    assert best_patterns[-1][1] == best_score

    # The top 9 are pretty stable
    correct_result = [(610, 21931),
                      (763, 29506),
                      (718, 32113),
                      (915, 33083),
                      (468, 33445),
                      (352, 33570),
                      (567, 33609),
                      (786, 33629),
                      (416, 33641),
                      (933, 33647),
                      (1222, 33652),
                      (421, 33656),
                      (1097, 33659),
                      (191, 33661),
                      (961, 33663),
                      (358, 33664),
                      (436, 33665),
                      (296, 33666),
                      (888, 33667),
                      (490, 33668),
                      (351, 33669)]

    for i in range(0, 9):
        assert best_patterns[i][1] == correct_result[i][1]
        assert best_patterns[i][0] == correct_result[i][0]

    # This last test is a bit fragile - it is possible that a slightly
    # different ordering can lead to a different number of patterns
    # found
    assert(len(best_patterns) == len(correct_result))

    # Again, this is a fragile test as different ordering or different
    # selection of a duplicate could lead to a different final result.
    # We would hope that the scores are equal, however...
    for i in range(0, len(best_patterns)):
        assert best_patterns[i][1] == correct_result[i][1]


if __name__ == "__main__":
    try:
        test_minimal_markers()
        print("PASSED")
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nFAILED")
        import sys
        sys.exit(-1)
