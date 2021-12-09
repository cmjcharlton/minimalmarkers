
import pandas as pd

import sys

try:
    input_file = sys.argv[1]
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

# Replace any cells which aren't 0, 1 or 2 with "x" - This will
# convert typical bad/missing data values such as "-1" or "NaN" to "x"
# This is imporatant as the data are converted to a single string for
# each marker so there must be exactly one chracter per column.
def clean(x):
    try:
        x = int(x)
        if x >= 0 and x <= 2:
            return x
    except Exception:
        pass

    return "x"

df = df.applymap(
    lambda x: clean(x)
)

print(df)


