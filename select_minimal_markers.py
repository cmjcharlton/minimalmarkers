
import pandas as pd
import numpy as np

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

    return "x"


# Replace any cells which aren't 0, 1 or 2 with "x" - This will
# convert typical bad/missing data values such as "-1" or "NaN" to "x"
# This is imporatant as the data are converted to a single string for
# each marker so there must be exactly one chracter per column.
df = df.applymap(
    lambda x: clean(x)
)

pattern_to_id = {}

# process each row
for i in range(0, nrows):
    alleles = {}
    fails = 0

    row = df.iloc[i]
    counts = row.value_counts()

    for key in counts.keys():
        if key == "x":
            fails = int(counts["x"])
        else:
            alleles[key] = int(counts[key])

    n_alleles = len(alleles.keys())

    rowlen = len(row)

    call_rate = float(rowlen - fails) / rowlen

    if n_alleles > 1 and call_rate > min_call_rate:
        # turn the data into a single string that contains the pattern
        idx = df.index[i]
        pattern = "".join([str(x) for x in row])

        if pattern in pattern_to_id:
            print(f"WARNING: Duplicate pattern - {idx} "
                  f"vs. {pattern_to_id[pattern]}")

        pattern_to_id[pattern] = idx

n = len(pattern_to_id)

print(f"Loaded marker data for {n} distinct patterns")

# All done with reading data from the SNP input file now.

# Now loop over the distinct SNP patterns to organise them by Minor Allele
# Frequency (MAF) score. This part is only really relevant if
# maxmarkers is set to fewer than the actual number of input markers,
# otherwise all get used anyway regardless of MAF ordering.

keys = list(pattern_to_id.keys())
keys.sort()

pattern_to_maf = {}
order_by_maf = {}

for pattern in keys:
    idx = pattern_to_id[pattern]
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

n = len(selected)

print(f"{n} distinct SNP patterns selected for constructing the optimal "
      "dataset")

# create the scoring matrix
matrix = np.zeros((ncols, ncols))

print(f"Built initial scoring matrix: {matrix.shape}")

cumulative = 0
target_size = int((ncols * (ncols-1)) / 2)

print(f"{target_size} varietal comparisons")

current_score = 1.0

print("\nIteration\tCumulativeResolved\tProportion\tMarkerID\tPattern")
