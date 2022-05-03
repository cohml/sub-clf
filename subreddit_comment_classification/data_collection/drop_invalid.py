"""
Remove bad rows from .parquet files of scraped comment data, writing remainder to new
.parquet files with "_cleaned" appended to the filename.
"""


import argparse
import pandas as pd
import re

from numpy import nan
from pathlib import Path
from typing import Tuple

from ..utils.defaults import DEFAULTS
from ..utils.misc import full_path


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

    # comments with a stray quotation mark screw up the pandas parser, but this is
    # hard to fix surgically, so just be safe and nuke all quotation marks
    df = (df.replace(r'["“”‟„⹂〞〟＂❝❞]', '', regex=True)
            .replace('', nan))  # b/c comments that were entirely quotation marks are now blank
    # drop rows with NaN entries in any of the following columns
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

    # drop rows whose entries in certain columns are too long or short, or missing
    for column, length in {'subreddit_id' : 8, 'id' : 7, 'link_id' : 9}.items():
        df = df.loc[df[column].notna() & df[column].str.len().eq(length)]

    nrows_final = df.shape[0]

    return df, nrows_original, nrows_final


def normalize_dtypes_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert individual data values to ensure that no columns contain mixed data types.

    Parameters
    ----------
    df : pd.DataFrame
        all valid data scraped for a single subreddit

    Returns
    -------
    df : pd.DataFrame
        all valid data scraped for a single subreddit, with data types normalized
    """

    # the following columns either (1) should only have NaN values, (2) are totally
    # uninteresting to me, or (3) keep introducing mixed data type issues down the
    # line, so I just keep things simple and remove them here
    df = df.drop(columns=['approved_at_utc',
                          'approved_by',
                          'awarders',
                          'author_cakeday',
                          'author_flair_richtext',
                          'author_flair_template_id',
                          'author_fullname',
                          'author_is_blocked',
                          'author_patreon_flair',
                          'associated_award',
                          'banned_at_utc',
                          'banned_by',
                          'body_html',
                          'can_mod_post',
                          'collapsed_because_crowd_control',
                          'collapsed_reason_code',
                          'comment_type',
                          'editable',
                          'likes',
                          'mod_note',
                          'mod_reason_by',
                          'mod_reports',
                          'num_reports',
                          'removal_reason',
                          'report_reasons',
                          'saved',
                          'top_awarded_type',
                          'unrepliable_reason',
                          'user_reports'], errors='ignore')

    # convert all values in select int columns to their integer counterparts, else to 0
    cols = ['gilded', 'total_awards_received']
    df[cols] = df[cols].applymap(lambda x: int(x) if str(x).isdigit() else 0)
    df[cols] = df[cols].astype(int)

    # ensure that NaN entries in boolean columns are all `False`
    bool_cols = ['archived', 'author_premium', 'can_gild', 'locked', 'mod_reason_title',
                 'score_hidden', 'send_replies', 'stickied']
    for bool_col in bool_cols:
        df.loc[df[bool_col].isna(), bool_col] = False
        df.loc[df[bool_col] == 'False', bool_col] = False
        df.loc[df[bool_col] == 'True', bool_col] = True

    # deal with other boolean columns that may require special handling
    for bool_col in ['collapsed', 'no_follow', 'is_submitter']:
        df.loc[df[bool_col] == 'True', bool_col] = True
        df.loc[df[bool_col] != True, bool_col] = False
        df[bool_col] = df[bool_col].astype(bool)

    # for select str columns, set NaN entries "unknown"
    for col in ['author_flair_text_color', 'author_flair_background_color',
                'distinguished', 'name', 'permalink']:
        df.loc[df[col].isna(), col] = 'unknown'
        df[col] = df[col].astype(str)

    # for select str columns, set NaN entries to "none"
    for col in ['author_flair_css_class', 'author_flair_text']:
        df.loc[df[col].isna(), col] = 'none'
        df[col] = df[col].astype(str)

    # for select str cols, set non-str entries (e.g, NaN, [], False) to 'not_applicable'
    str_cols = ['author_flair_type', 'collapsed_reason']
    for str_col in str_cols:
        df.loc[df[str_col].isna(), str_col] = 'not_applicable'
        df.loc[df[str_col] == '[]', str_col] = 'not_applicable'
        df.loc[df[str_col] == 'False', str_col] = 'not_applicable'
        df.loc[df[str_col].apply(type) != str, str_col] = 'not_applicable'
        df[str_col] = df[str_col].astype(str)

    # for select int cols, convert NaN entries to 0
    for col in ['score', 'ups']:
        df.loc[df[col].isna(), col] = 0
        df[col] = df[col].astype(float).astype(int)  # just to be safe, lest decimals

    # set NaN entries in `controversiality` to 0, as this appears to be a binary column
    df.loc[df.controversiality.isna(), 'controversiality'] = 0
    df.controversiality = df.controversiality.astype(float).astype(int)

    # set `downs` to 0, since this field has been deprecated by Reddit
    df.downs = 0
    df.downs = df.downs.astype(int)

    # convert non-numeric entries in `edited` to NaN
    df.loc[df.edited.isin(['True', True, 'False', False, '[]']), 'edited'] = nan
    df.edited = df.edited.astype(float)

    # set `gildings` to "{}" for all entries with no listed guildings
    df.loc[~df.gildings.astype(str).str.contains("'gid_"), 'gildings'] = '{}'

    # set `subreddit_name_prefixed` to be "r/" + `subreddit`, as intended I think
    df.subreddit_name_prefixed = 'r/' + df.subreddit
    df.subreddit_name_prefixed = df.subreddit_name_prefixed.astype(str)

    # set `subreddit_type` to "public" globally, since only public subs can be scraped
    df.subreddit_type = 'public'
    df.subreddit_type = df.subreddit_type.astype(str)

    # set NaN entries in `treatment_tags` to string representation of empty list
    df.loc[df.treatment_tags.isna(), 'treatment_tags'] = '[]'
    df.treatment_tags = df.treatment_tags.astype(str)

    return df


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
                    'new .parquet file with the same name as the source plus the '
                    'suffix "_cleaned".'
    )
    parser.add_argument(
        '-i', '--input-directory',
        type=full_path,
        default=DEFAULTS['PATHS']['DIRS']['ALL_FIELDS'],
        help='path to directory with .parquet files (one per subreddit) containing '
             'all fields scraped from Reddit (default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--output-directory',
        type=full_path,
        help='path to directory to write output files to (one per subreddit); if '
             'unspecified, defaults to the input directory (default: %(default)s)'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # set output directory, and if needed, create it
    output_directory = args.output_directory or args.input_directory
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
        print('created:', output_directory)

    safe_parsing_params = {'dtype' : object,
                           'engine' : 'pyarrow',
                           'on_bad_lines' : 'skip'}

    for subreddit in args.input_directory.glob('*.parquet'):

        if subreddit.stem.endswith('_cleaned'):
            continue

        print('-' * 100)
        print('reading:', subreddit.name)
        df = pd.read_parquet(subreddit, **safe_parsing_params)

        df, nrows_original, nrows_final = drop_invalid_rows(df)
        print('dropped:', f'{nrows_original - nrows_final:,} rows '
              f'({(nrows_original - nrows_final) / nrows_original:.2%})')

        # ensure no columns contain mixed data types
        df = normalize_dtypes_by_column(df)

        output_filename = output_directory / (subreddit.stem + '_cleaned.parquet')
        df.to_parquet(output_filename, index=False)
        print('written:', output_filename)


if __name__ == '__main__':
    main()
