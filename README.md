# minimalmarkers

Code for choosing the minimum set of genetic markers needed to differentiate all samples in a genotyping dataset

## Requirements

* Python >= 3.7
* numpy >= 1.20.0
* numba >= 0.50.0

Optional (to display progress bars)

* tqdm >= 4.60.0

## Usage

```
python select_minimal_markers.py <input_file>
```

where `<input_file>` is an input file in genotypes format. If your input
file is in VCF format then use

```
python select_minimal_markers.py --vcf <input_file>
```

## Example

The sample SeqSNP data for Cider apples is included as a test dataset
in the `example` directory. To run this, type;

```
python select_minimal_markers.py example/AppleGenotypes.csv
```

The script will first pick the marker which discriminates the maximum
number of varieties (iteration 1).

For iteration 2, it will find the marker which discriminates the maximum
varieties which were NOT discriminated by the marker in iteration 1.

This process continues until either;

1. there are no markers left,
2. adding more markers doesn't add additional varietal discrimination, or,
3. all varieties are discriminated.

Output is written to tab delimited text files (called
`{input_file}_minimal_markers.txt` and
`{input_file}_selected_markers_with_headers.txt}`) which can be
viewed in Excel or similar.

The Apple example data supplied has 1286 markers and 260 varieties
and runs in about a second. Larger datasets may take much longer!

Example output for the Cider apples data set is included in the
`example/example_output` directory.

For the example data, you should find that 23 SNPs will discriminate
all varieties except "Willy" versus "Connie" - which are not distinguishable.

A more detailed explanation of the approach can be found in
our paper: https://doi.org/10.1371/journal.pone.0242940

## Testing and validation

The `select_minimal_markers.py` script contains several runtime validation
and sanity tests that check that it works correctly every time
that it runs. The most important validation is that, after finding the
set of minimal markers, it rebuilds the selection matrix from scratch using
those markers. It validates that the scores calculated during the rebuild
match those found during the search, and then validates that the number
of varieties that are distinguished matches the maximum number that could
be distinguished if all markers were selected. Warnings will be printed
to the output if any of these checks fail.

You can run an integration test to validate installation by typing;

```
pytest .
```

(or you can type `python test_minimal_markers.py` if you don't have
`pytest` installed)

## Genotypes file format

The example data file uses a genotypes file format. This is comma separated
and uses 0 for AA, 1 for AB and 2 for BB. It will also accept tab separated
data and A, AB, B formatted genotype calls as input.

Note that you can read in data in VCF format by using the `--vcf` command
line option. This may not work for all VCFs, so please check any warnings
that are printed and report any bugs.

## History

This script is a rewrite of the original `select_minimal_markers.pl` Perl
script that was reported in the above paper. The rewrite optimises the code
and improves ease of use and robustness. The original Perl scripts
associated with the paper are included in the `original` directory.
These include the original `convert_vcf_to_genotypes.pl` and
`check_results.pl` scripts that convert VCF files and check the output
of the code. The functionality of these scripts have been merged into
`select_minimal_markers.py`.
