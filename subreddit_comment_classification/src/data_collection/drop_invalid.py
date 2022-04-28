"""
Remove bad rows from CSV files of scraped comment data, writing remainder to new CSV
files with "_cleaned" appended to the filename.
"""


import argparse
import pandas as pd
import re

from pathlib import Path
from typing import Tuple

from subreddit_comment_classification.src.utils.const import DEFAULTS


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description='Apply assumptions and rule-based heuristics to identify and drop '
                    'invalid samples from data set, writing the cleaned results to a '
                    'new CSV file with the same name as the source plus "_cleaned".'
    )
    parser.add_argument(
        '-i', '--input-directory',
        type=lambda s: Path(s).resolve(),
        default=DEFAULTS['PATHS']['DIRS']['ALL_FIELDS'],
        help='path to directory with CSV files (one per subreddit) containing all '
             'fields scraped from Reddit (default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--output-directory',
        type=lambda s: Path(s).resolve(),
        help='path to directory to write output files to (one per subreddit); if '
             'unspecified, defaults to the input directory (default: %(default)s)'
    )
    return parser.parse_args()


def drop_invalid_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Check all rows and remove those which meet any of the following conditions:

    1. NaN entry in any of the following columns:
        {'author', 'body', 'id', 'parent_id', 'subreddit', 'subreddit_id'}

    2. Comment was deleted by the OP or removed by a moderator

    3. Name of subreddit or author contains illegal characters or is too long/short

    4. Entry in any of the following columns is too long/short:
        {'id', 'link_id', 'subreddit_id'}

    Parameters
    ----------
    df : pd.DataFrame
        all data scraped for a single subreddit

    Returns
    -------
    df : pd.DataFrame
        all data scraped for a single subreddit, minus any rows identified as corrupt
        or invalid
    """

    nrows_original = df.shape[0]

    # drop rows with non-NaN entries in columns which should be NaN
    non_nan_columns = ['author', 'body', 'id', 'parent_id', 'subreddit', 'subreddit_id']
    df = df.loc[df[non_nan_columns].notna().all(axis=1)]

    # drop rows whose bodies were deleted by mods or OP
    df = df.loc[~df.body.isin(['[deleted]', '[removed]'])]

    # drop rows whose author/subreddit name has illegal chars or is too long/short
    valid_author_name = re.compile(r'^[\w-]{3,20}$')
    author_is_valid = df.author.str.match(valid_author_name)
    valid_subreddit_name = re.compile(r'^[A-Za-z0-9]\w{2,20}$')
    subreddit_is_valid = df.subreddit.str.match(valid_subreddit_name)
    df = df.loc[author_is_valid & subreddit_is_valid]

    # drop rows whose entries in certain columns are too long or short
    for column, length in {'subreddit_id' : 8, 'id' : 7, 'link_id' : 9}.items():
        df = df.loc[df[column].str.len() == length]

    nrows_final = df.shape[0]

    return df, nrows_original, nrows_final


def main() -> None:
    args = parse_args()

    # set output directory, and if needed, create it
    output_directory = args.output_directory or args.input_directory
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
        print('created:', output_directory)

    safe_parsing_params = {'dtype' : str,
                           'engine' : 'python',
                           'on_bad_lines' : 'skip'}

    for subreddit in args.input_directory.glob('*.csv'):

        if subreddit.stem.endswith('_cleaned'):
            continue

        print('-' * 100)
        print('reading:', subreddit.name)
        df = pd.read_csv(subreddit, **safe_parsing_params)

        df, nrows_original, nrows_final = drop_invalid_rows(df)
        print('dropped:', f'{nrows_original - nrows_final:,} rows '
              f'({(nrows_original - nrows_final) / nrows_original:.2%})')

        output_filename = output_directory / (subreddit.stem + '_cleaned.csv')
        df.to_csv(output_filename, index=False)
        print('written:', output_filename)


if __name__ == '__main__':
    main()
