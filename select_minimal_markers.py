
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
max_markers = 1000000000000

# Ignore markers with less than this proportion of valid (0, 1 of 2 ) calls.
# (Bad calls might be encoded as -1 or NaN)
min_call_rate = 0.9

# MAF is minor allele frequency.  This is set to a low level to include as
# many markers as possible but exclude rare error calls.
# It probably needs optimising for your data.
min_maf = 0.001

# Get the filename from the command line
try:
    input_file = sys.argv[1]
except Exception:
    print("USAGE: python select_minimal_markers.py genotypes.csv")
    sys.exit(0)

# Read in the data - assume this is comma separated for now. Can
# easily add a test to change to tab separated if needed
df = pd.read_csv(input_file, index_col="code")

# get the number of columns - this should be the same for every row
ncols = df.shape[1]
nrows = df.shape[0]

# Common formats are A, B for homozygotes (AA and BB) and AB for
# heterozygotes or 0,1,2 for hom AA, het AB and hom BB.
# We're going to convert A AB B to 0,1,2 format from here on
df = df.replace("AB", 1).replace("A", 0).replace("B", 2)


def clean(x):
    try:
        x = int(x)
        if x >= 0 and x <= 2:
            return x
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


def get_pattern_from_array(array):
    """Return a pattern encoded as an integer array
       into a string - this converts values less than
       0 into x
    """
    pattern = [str(x) if x >= 0 else "x" for x in array]
    return "".join(pattern)


pattern_to_idx = {}

# process each row
for i in range(0, nrows):
    alleles = {}
    fails = 0

    row = df.iloc[i]
    counts = row.value_counts()

    for key in counts.keys():
        if key == -1:
            fails = int(counts[-1])
        else:
            alleles[key] = int(counts[key])

    n_alleles = len(alleles.keys())

    rowlen = len(row)

    call_rate = float(rowlen - fails) / rowlen

    if n_alleles > 1 and call_rate > min_call_rate:
        # turn the data into a single string that contains the pattern
        idx = df.index[i]
        pattern = get_pattern_from_array(row)

        if pattern in pattern_to_idx:
            print(f"WARNING: Duplicate pattern - {idx} "
                  f"vs. {pattern_to_idx[pattern]}")

        pattern_to_idx[pattern] = idx

n = len(pattern_to_idx)

print(f"Loaded marker data for {n} distinct patterns")

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
    zero = pattern.count("0")
    one = pattern.count("1")
    two = pattern.count("2")

    # Logic steps to work out which is the second most common
    # call, which we'll define as the minor allele.
    if one >= zero and zero >= two:
        minor = zero
    elif zero >= one and one >= two:
        minor = one
    elif zero >= two and two >= one:
        minor = two
    elif one >= two and two >= zero:
        minor = two
    elif two >= one and one >= zero:
        minor = one
    elif two >= zero and zero >= one:
        minor = zero
    else:
        print(f"WARNING: MISSED CONDITION!")
        sys.exit(-1)

    maf = float(minor) / ncols
    # print(f"{idx}\t{maf}\t{zero}\t{one}\t{two}\n")

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


@numba.jit(nopython=True, fastmath=True, nogil=True)
def score_pattern(pattern, matrix):
    n = len(pattern)

    # The logic here is to loop through the genotype string for this
    # marker row and compare each position with every position to
    # the right of it
    # This will fill out one triangular half of the matrix so we end
    # up comparing col1 with col2 then col3 but we don't waste time
    # going back and comparing col2 to col1.
    score = 0

    for i in numba.prange(0, n):
        ichar = pattern[i]

        if ichar != "x":
            for j in range(i+1, n):
                jchar = pattern[j]

                # If this cell in the matrix is currently set to zero,
                # i.e. this pair of varieties (i and j) are unresolved,
                # and their genotypes are valid and different, then we can
                # set this cell in the test matrix to 1 (= resolved) -
                # otherwise it remains set to zero.
                score += (jchar != "x" and ichar != jchar and
                          matrix[i, j] == 0)

    return score


@numba.jit(nopython=True, fastmath=True, nogil=True)
def score_patterns(patterns, matrix):
    best_score = 0
    best_pattern = None

    npatterns = len(patterns)

    for i in range(0, npatterns):
        pattern = patterns[i]
        score = score_pattern(pattern, matrix)

        # If the current marker is better than others tested, it becomes the
        # new bestscore and its matrix becomes the one to beat!
        if score > best_score:
            best_score = score
            best_pattern = pattern

    return (best_score, best_pattern)


def create_matrix(pattern, matrix):
    n = len(pattern)

    m = np.zeros((n, n), np.int8)

    for i in range(0, n):
        ichar = pattern[i]

        if ichar != "x":
            for j in range(i+1, n):
                jchar = pattern[j]

                # If this cell in the matrix is currently set to zero,
                # i.e. this pair of varieties (i and j) are unresolved,
                # and their genotypes are valid and different, then we can
                # set this cell in the test matrix to 1 (= resolved) -
                # otherwise it remains set to zero.
                if jchar != "x" and ichar != jchar and matrix[i, j] == 0:
                    m[i, j] = 1

    return m


n = len(selected)

print(f"{n} distinct SNP patterns selected for constructing the optimal "
      "dataset")

# create the scoring matrix
matrix = np.zeros((ncols, ncols), np.int8)

print(f"Built initial scoring matrix: {matrix.shape}")

cumulative = 0
target_size = int((ncols * (ncols-1)) / 2)

print(f"{target_size} varietal comparisons")

current_score = 1

print("\nIteration\tCumulativeResolved\tProportion\tMarkerID\tPattern")

iteration = 0


# This is the main loop where we iterate through all of the available rows of
# SNP data and find the one that adds the most new "1s" to the overall
# scoring matrix
# This loop will exit when the current_score value is zero - i.e. adding
# another row doesn't add anything to the overall matrix score
while current_score > 0:
    # This hash is the current working score matrix - it will be evaluated
    # for this iteration and its contents added to the overal matrix
    # once we have decided which SNP row is best for this iteration
    best_matrix = np.zeros((ncols, ncols), np.int8)
    best_score = 0
    iteration += 1

    (best_score, best_pattern) = score_patterns(selected, matrix)

    idx = pattern_to_idx[best_pattern]

    print(f"best score is {best_score}")

    if best_score > 0:
        cumulative += best_score
        proportion_resolved = cumulative / target_size
        resolved = "%.6f" % proportion_resolved

        print(f"{iteration}\t{cumulative}\t{resolved}\t{idx}\t"
              f"{best_pattern}")

    current_score = best_score

    matrix += create_matrix(best_pattern, matrix)

    best_score = 0.0
    best_matrix = None
