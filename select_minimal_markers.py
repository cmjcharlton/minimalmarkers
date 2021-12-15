
import numba as _numba
import numpy as _np


def get_pattern_from_array(array) -> str:
    """Return a pattern encoded as an integer array
       into a string - this converts values less than
       0 into x
    """

    # Common formats are A, B for homozygotes (AA and BB) and AB for
    # heterozygotes or 0,1,2 for hom AA, het AB and hom BB.
    # We're going to convert A AB B to 0,1,2 format from here on

    values = {"0": "0",
              "1": "1",
              "2": "2",
              0: "0",
              1: "1",
              2: "2",
              "AB": "1",
              "A": "0",
              "B": "2"}

    pattern = [values.get(x, "x") for x in array]
    return "".join(pattern)


def load_patterns(filename: str,
                  min_call_rate: float = 0.9,
                  print_progress: bool = False):
    """Load all of the patterns from the passed file.
       The patterns will be converted to the correct format,
       including cleaning / conversion of A, B, AB converted
       to 0, 1, 2 format.

       The patterns with a poor call_rate will be removed.

       min_call_rate: Ignore markers with less than this proportion
                      of valid (0, 1 of 2 ) calls.
                      (Bad calls might be encoded as -1 or NaN)

       This will return a dictionary of unique patterns.
       The keys will be the patterns, and the values
       are the IDs of the patterns.
    """
    import pandas as pd

    # Read in the data - assume this is comma separated for now. Can
    # easily add a test to change to tab separated if needed
    if print_progress:
        print(f"Loading '{input_file}'...")

    df = pd.read_csv(input_file, index_col="code")

    # process each row
    if print_progress:
        print(f"Data read! Processing rows...")

    nrows: int = df.shape[0]
    patterns = {}
    duplicates = {}
    rowlen: int = -1

    from tqdm import tqdm

    for i in tqdm(range(0, nrows)):
        alleles = {}
        fails: int = 0

        pattern = get_pattern_from_array(df.iloc[i])

        if pattern in patterns:
            if pattern in duplicates:
                duplicates[pattern].append(df.index[i])
            else:
                duplicates[pattern] = [df.index[i]]
            next

        fails: int = pattern.count("x")

        n_alleles = 0

        for allele in ["0", "1", "2"]:
            if pattern.count(allele) > 0:
                n_alleles += 1

        if rowlen == -1:
            rowlen = len(pattern)
        else:
            if rowlen != len(pattern):
                print(f"WARNING: Wrong rowlen! {len(pattern)}")

        call_rate: float = float(rowlen - fails) / rowlen

        if n_alleles > 1 and call_rate > min_call_rate:
            patterns[pattern] = df.index[i]

    if print_progress:
        print(f"\nLoaded marker data for {len(patterns)} distinct patterns")

    return patterns


def sort_and_filter_patterns(patterns,
                             max_markers: int = 1000000000000,
                             min_maf: float = 0.001,
                             print_progress: bool = False):
    """Now loop over the distinct SNP patterns to organise them by
       Minor Allele Frequency (MAF) score.

       This will convert the dictionary of patterns into
       a 2D integer array (rows = sorted patterns, columns are
       the -1, 0, 1, 2 values for each pattern).

       max_markers: This is normally set to more than the number of markers
                    in the input file (~ 35000) but if it is set lower then
                    markers are prioritised.
                    e.g. if set to 5000, then the top 5000 markers by MAF
                    score will be used and the rest ignored. You will get a
                    warning if this limit is reached.
                    This feature is intended to speed up the runtime for
                    really big datasets such as 800K Axiom genotyping.

        min_maf: MAF is minor allele frequency.  This is set to a low level
                 to include as many markers as possible but exclude rare
                 error calls. It probably needs optimising for your data.

       This will return the matrix plus a list of pattern IDs
       (in the same order as the matrix)
    """
    keys = list(patterns.keys())
    keys.sort()

    pattern_to_maf = {}
    order_by_maf = {}

    for pattern in keys:
        zero: int = pattern.count("0")
        one: int = pattern.count("1")
        two: int = pattern.count("2")

        # Logic steps to work out which is the second most common
        # call, which we'll define as the minor allele.
        if one >= zero and zero >= two:
            minor: int = zero
        elif zero >= one and one >= two:
            minor: int = one
        elif zero >= two and two >= one:
            minor: int = two
        elif one >= two and two >= zero:
            minor: int = two
        elif two >= one and one >= zero:
            minor: int = one
        elif two >= zero and zero >= one:
            minor: int = zero
        else:
            raise ValueError("Invalid condition!")

        maf: float = float(minor) / len(pattern)

        if maf > min_maf:
            pattern_to_maf[pattern] = maf

            if maf not in order_by_maf:
                order_by_maf[maf] = []

            order_by_maf[maf].append(pattern)

    if print_progress:
        print("Sorting by minor allele frequency..")

    scores = list(order_by_maf.keys())
    scores.sort()

    selected = []

    for score in scores:
        if len(selected) < max_markers:
            for pattern in order_by_maf[score]:
                if len(selected) < max_markers:
                    selected.append(pattern)
                else:
                    print(f"Maximum marker count {max_markers} reached! "
                          "Ignoring further markers.")

    # now sort the selected markers by pattern
    # selected.sort()

    # copy the patterns into a numpy array
    pattern_matrix = _np.zeros((len(selected), len(selected[0])), _np.int8)
    pattern_ids = []

    from tqdm import tqdm

    for i in tqdm(range(0, len(selected))):
        pattern = selected[i]
        pattern_ids.append(patterns[pattern])

        for j in range(0, len(pattern)):
            x = pattern[j]

            if x == "x":
                pattern_matrix[i, j] = -1
            else:
                pattern_matrix[i, j] = x

    return (pattern_matrix, pattern_ids)


@_numba.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def score_patterns(patterns, matrix, skip_patterns):
    """Do the work of scoring all of the passed patterns against
       the current value of the matrix. This returns a tuple
       of the best score and the index of the pattern with
       that best score
    """
    npatterns: int = patterns.shape[0]
    ncols: int = patterns.shape[1]

    scores = _np.zeros(npatterns, _np.int32)

    for p in _numba.prange(0, npatterns):
        if not skip_patterns[p]:
            score: int = 0

            for i in range(0, ncols):
                ival: int = patterns[p, i]

                if ival != -1:
                    for j in range(i+1, ncols):
                        jval: int = patterns[p, j]

                        score += (jval != -1 and ival != jval and
                                  matrix[i, j] == 0)

            scores[p] = score

            if score == 0:
                skip_patterns[p] = 1

    best_score: int = 0
    best_pattern: int = 0

    for i in range(0, npatterns):
        if scores[i] > best_score:
            best_score = scores[i]
            best_pattern = i

    skip_patterns[best_pattern] = 1

    return (best_score, best_pattern)


@_numba.jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def create_matrix(pattern, matrix):
    """Create the stencil matrix for this pattern, based on the
       current value of the combined stencil matrix (i.e. only
       cover up holes that are not already covered
    """
    n = len(pattern)

    m = _np.zeros((n, n), _np.int8)

    for i in _numba.prange(0, n):
        ival = pattern[i]

        if ival != -1:
            for j in range(i+1, n):
                jval = pattern[j]

                # If this cell in the matrix is currently set to zero,
                # i.e. this pair of varieties (i and j) are unresolved,
                # and their genotypes are valid and different, then we can
                # set this cell in the test matrix to 1 (= resolved) -
                # otherwise it remains set to zero.
                if jval != -1 and ival != jval and matrix[i, j] == 0:
                    m[i, j] = 1

    return m


def find_best_patterns(patterns, pattern_ids,
                       print_progress: bool = False):
    """This is the main function where we iterate through all of the
       available rows of SNP data and find the one that adds the most
       new "1s" to the overall scoring matrix. This function will return
       when the current_score value is zero - i.e. adding
       another row doesn't add anything to the overall matrix score.

       This return the sorted list of best patterns, plus the
       matrix showing what is being distinguished
    """

    # create the scoring matrix
    ncols: int = len(patterns[0])

    perfect_score: int = int((ncols * (ncols-1)) / 2)

    if print_progress:
        print(f"{perfect_score} varietal comparisons")
        print("\nIteration\tCumulativeResolved\tProportion\tMarkerID\tPattern")

    matrix = _np.zeros((ncols, ncols), _np.int8)

    iteration: int = 0
    cumulative_score: int = 0
    current_score: int = 1

    best_patterns = []

    skip_patterns = _np.zeros(len(patterns))

    while current_score > 0:
        iteration += 1

        (best_score, best_pattern) = score_patterns(patterns, matrix,
                                                    skip_patterns)

        if best_score > 0:
            cumulative_score += best_score
            proportion_resolved = cumulative_score / perfect_score
            best_patterns.append(best_pattern)
            matrix += create_matrix(patterns[best_pattern], matrix)

            if print_progress:
                pattern = get_pattern_from_array(patterns[best_pattern])
                pattern_id = pattern_ids[best_pattern]

                print(f"{iteration}\t{cumulative_score}\t"
                      f"{proportion_resolved:.6f}\t"
                      f"{pattern_id}\t{pattern}")

        current_score: int = best_score

    return (best_patterns, matrix)


if __name__ == "__main__":
    # Get the filename from the command line
    try:
        import sys
        input_file: str = sys.argv[1]
    except Exception:
        print("USAGE: python select_minimal_markers.py genotypes.csv")
        sys.exit(0)

    patterns = load_patterns(input_file, print_progress=True)

    print(f"{len(patterns)} distinct SNP patterns selected for "
          "constructing the optimal dataset")

    (patterns, pattern_ids) = sort_and_filter_patterns(patterns,
                                                       print_progress=True)

    (best_patterns, matrix) = find_best_patterns(patterns,
                                                 pattern_ids,
                                                 print_progress=True)
