"""
Read in one or more .csv files containing all fields scraped from Reddit,
extract the comment IDs and bodies from each, and write them to a new .csv
with the same file basename as the source all-fields .csv.
"""


import argparse
import pandas as pd

from pathlib import Path

from const import DEFAULTS


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
        parsed command line arguments
    """

    parser = argparse.ArgumentParser(
        description='Pull text out of raw data extract.'
    )
    csvs = parser.add_mutually_exclusive_group()
    csvs.add_argument(
        '-f', '--all-fields-file',
        type=lambda p: Path(p).resolve(),
        required=False,
        help='path to "all-fields" .csv to process; if not specified, all '
             '"all-fields" .csv\'s will be processed'
    )
    csvs.add_argument(
        '-d', '--all-fields-directory',
        type=lambda p: Path(p).resolve(),
        default=DEFAULTS['PATHS']['DIRS']['ALL_FIELDS'],
        required=False,
        help='path to directory containing "all-fields" .csvs (default: '
             '%(default)s)'
    )
    parser.add_argument(
        '-o', '--output-directory',
        type=lambda p: Path(p).resolve(),
        default=DEFAULTS['PATHS']['DIRS']['BODY_ONLY'] / 'raw_text',
        required=False,
        help='path to directory to write all output files (default %(default)s)'
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all_fields_file:
        all_fields_files = args.all_fields_file
    else:
        all_fields_files = args.all_fields_directory.glob('*.csv')
    all_fields_files = sorted(all_fields_files, key=lambda p: p.name.lower())

    read_csv_params = dict(usecols=['id', 'body'],
                           index_col='id',
                           dtype=str)

    for all_fields_file in all_fields_files:
        output_file = args.output_directory / all_fields_file.name
        pd.read_csv(all_fields_file, **read_csv_params).to_csv(output_file)
        print('written:', output_file)


if __name__ == '__main__':
    main()
