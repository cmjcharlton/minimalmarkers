
import pandas as pd
import numpy as np
import numba

import sys

from pandas.core import indexing

# Here are the default values of key parameters

# This is normally set to more than the number of markers in the input file
# (~ 35000) but if it is set lower then markers are prioritised.
# e.g. if set to 5000, then the top 5000 markers by MAF score will be used
# and the rest ignored. You will get a warning if this limit is reached.
# This feature is intended to speed up the runtime for really big datasets
# such as 800K Axiom genotyping.
max_markers: int = 1000000000000

# Ignore markers with less than this proportion of valid (0, 1 of 2 ) calls.
# (Bad calls might be encoded as -1 or NaN)
min_call_rate: float = 0.9

# MAF is minor allele frequency.  This is set to a low level to include as
# many markers as possible but exclude rare error calls.
# It probably needs optimising for your data.
min_maf: float = 0.001

# Get the filename from the command line
try:
    input_file: str = sys.argv[1]
except Exception:
    print("USAGE: python select_minimal_markers.py genotypes.csv")
    sys.exit(0)

# Read in the data - assume this is comma separated for now. Can
# easily add a test to change to tab separated if needed
df = pd.read_csv(input_file, index_col="code")

# Common formats are A, B for homozygotes (AA and BB) and AB for
# heterozygotes or 0,1,2 for hom AA, het AB and hom BB.
# We're going to convert A AB B to 0,1,2 format from here on
df = df.replace("AB", 1).replace("A", 0).replace("B", 2)


def clean(x: any) -> int:
    """Quick cleaning function that returns 0, 1, 2 if x is
       0, 1 or 2, or returns -1 if it is any other value
    """

    try:
        xi: int = int(x)
        if xi >= 0 and xi <= 2:
            return xi
    except Exception:
        pass

    return -1


# Replace any cells which aren't 0, 1 or 2 with -1 - This will
# convert typical bad/missing data values such as "NaN" to -1
# This is imporatant as the data are converted to a single string for
# each marker so there must be exactly one chracter per column.
df = df.applymap(
    lambda x: clean(x)
)


def get_pattern_from_array(array) -> str:
    """Return a pattern encoded as an integer array
       into a string - this converts values less than
       0 into x
    """
    pattern = [str(x) if x >= 0 else "x" for x in array]
    return "".join(pattern)


pattern_to_idx = {}

# process each row
nrows: int = df.shape[0]

for i in range(0, nrows):
    alleles = {}
    fails: int = 0

    row = df.iloc[i]
    counts = row.value_counts()

    for key in counts.keys():
        if key == -1:
            fails: int = int(counts[-1])
        else:
            alleles[key] = int(counts[key])

    n_alleles: int = len(alleles.keys())

    rowlen: int = len(row)

    call_rate: float = float(rowlen - fails) / rowlen

    if n_alleles > 1 and call_rate > min_call_rate:
        # turn the data into a single string that contains the pattern
        idx = df.index[i]
        pattern = get_pattern_from_array(row)

        if pattern in pattern_to_idx:
            print(f"WARNING: Duplicate pattern - {idx} "
                  f"vs. {pattern_to_idx[pattern]}")

        pattern_to_idx[pattern] = idx

print(f"Loaded marker data for {len(pattern_to_idx)} distinct patterns")

# All done with reading data from the SNP input file now.

# Now loop over the distinct SNP patterns to organise them by Minor Allele
# Frequency (MAF) score. This part is only really relevant if
# maxmarkers is set to fewer than the actual number of input markers,
# otherwise all get used anyway regardless of MAF ordering.

keys = list(pattern_to_idx.keys())
keys.sort()

pattern_to_maf = {}
order_by_maf = {}

for pattern in keys:
    idx = pattern_to_idx[pattern]
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
        print(f"WARNING: MISSED CONDITION!")
        sys.exit(-1)

    maf: float = float(minor) / len(pattern)

    if maf > min_maf:
        pattern_to_maf[pattern] = maf

        if maf not in order_by_maf:
            order_by_maf[maf] = []

        order_by_maf[maf].append(pattern)

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
selected.sort()

# copy the patterns into a numpy array
patterns = np.zeros((len(selected), len(selected[0])), np.int8)

for i in range(0, len(selected)):
    pattern = selected[i]

    for j in range(0, len(pattern)):
        x = pattern[j]

        if x == "x":
            patterns[i, j] = -1
        else:
            patterns[i, j] = x


@numba.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def score_patterns(patterns, matrix):

    npatterns: int = patterns.shape[0]
    ncols: int = patterns.shape[1]

    scores = np.zeros(npatterns, np.int32)

    for p in numba.prange(0, npatterns):
        score: int = 0

        for i in range(0, ncols):
            ival: int = patterns[p, i]

            if ival != -1:
                for j in range(i+1, ncols):
                    jval: int = patterns[p, j]

                    score += (jval != -1 and ival != jval and
                              matrix[i, j] == 0)

        scores[p] = score

    best_score: int = 0
    best_pattern: int = 0

    for i in range(0, npatterns):
        if scores[i] > best_score:
            best_score = scores[i]
            best_pattern = i

    return (best_score, best_pattern)


@numba.jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def create_matrix(pattern, matrix):
    n = len(pattern)

    m = np.zeros((n, n), np.int8)

    for i in numba.prange(0, n):
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


print(f"{len(selected)} distinct SNP patterns selected for "
      "constructing the optimal dataset")

# create the scoring matrix
ncols: int = len(selected[0])

perfect_score: int = int((ncols * (ncols-1)) / 2)

print(f"{perfect_score} varietal comparisons")


def find_best_patterns(patterns, print_progress: bool = True):
    """This is the main function where we iterate through all of the
       available rows of SNP data and find the one that adds the most
       new "1s" to the overall scoring matrix. This function will return
       when the current_score value is zero - i.e. adding
       another row doesn't add anything to the overall matrix score.

       This return the sorted list of best patterns, plus the
       matrix showing what is being distinguished
    """

    if print_progress:
        print("\nIteration\tCumulativeResolved\tProportion\tMarkerID\tPattern")

    matrix = np.zeros((ncols, ncols), np.int8)

    iteration: int = 0
    cumulative_score: int = 0
    current_score: int = 1

    best_patterns = []

    while current_score > 0:
        iteration += 1

        (best_score, best_pattern) = score_patterns(patterns, matrix)

        if best_score > 0:
            cumulative_score += best_score
            proportion_resolved = cumulative_score / perfect_score
            best_patterns.append(best_pattern)
            matrix += create_matrix(patterns[best_pattern], matrix)

            if print_progress:
                pattern = get_pattern_from_array(patterns[best_pattern])
                pattern_id = pattern_to_idx[pattern]

                print(f"{iteration}\t{cumulative_score}\t"
                      f"{proportion_resolved:.6f}\t"
                      f"{pattern_id}\t{pattern}")

        current_score: int = best_score

    return (best_patterns, matrix)


(best_patterns, matrix) = find_best_patterns(patterns)
