"""
Remove rows with any anomalous values from .parquet files of scraped comment data,
writing the validated remainder to new .parquet files in a new directory. The new
directory will have the same name as the source directory plus the suffix "_clean".
"""


import argparse
import pandas as pd
import re
import warnings
warnings.simplefilter("ignore", UserWarning)

from numpy import nan
from pathlib import Path
from typing import Tuple

from ..util.defaults import DEFAULTS
from ..util.io import load_raw_data
from ..util.misc import full_path


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

    nrows_original = len(df)

    # comments with a stray quotation mark screw up the pandas parser, but this is
    # hard to fix surgically, so just be safe and nuke all quotation marks
    df = (df.replace(r'["“”‟„⹂〞〟＂❝❞]', '', regex=True)
            .replace('', nan))  # b/c comments that were entirely quotation marks are now blank
    # drop rows with NaN entries in any of the following columns
    non_nan_columns = ['author', 'body', 'id', 'parent_id',
                       'post_id', 'subreddit', 'subreddit_id']
    df = df.loc[df[non_nan_columns].notnull().all(axis=1)]

    # drop rows whose bodies were deleted by mods or OP
    df = df.loc[~df.body.isin(['[deleted]', '[removed]'])]

    # drop rows whose author/subreddit name has illegal chars or is too long/short
    valid_author_name = re.compile(r'^[\w-]{3,20}$')
    author_is_valid = df.author.str.match(valid_author_name)
    valid_subreddit_name = re.compile(r'^[A-Za-z0-9]\w{2,20}$')
    subreddit_is_valid = df.subreddit.str.match(valid_subreddit_name)
    df = df.loc[author_is_valid & subreddit_is_valid]

    # drop rows whose entries in certain columns are too long or short, or missing
    lengths_by_column = {'id' : 7, 'link_id' : 9, 'post_id' : 6, 'subreddit_id' : 8}
    for column, length in lengths_by_column.items():
        df = df.loc[df[column].notnull() & df[column].str.len().eq(length)]

    nrows_final = len(df)

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
                          'media_metadata',
                          'mod_note',
                          'mod_reason_by',
                          'mod_reports',
                          'num_reports',
                          'removal_reason',
                          'report_reasons',
                          'saved',
                          'top_awarded_type',
                          'unrepliable_reason',
                          'user_reports'],
                 errors='ignore')

    # convert all values in select int columns to their integer counterparts, else to 0
    cols = ['gilded', 'total_awards_received']
    df[cols] = df[cols].applymap(lambda x: int(x) if str(x).isdigit() else 0)
    df[cols] = df[cols].astype(int)

    # ensure that NaN entries in boolean columns are all `False`
    bool_cols = ['archived', 'author_premium', 'can_gild', 'locked', 'mod_reason_title',
                 'score_hidden', 'send_replies', 'stickied']
    for bool_col in bool_cols:
        df[bool_col] = df[bool_col].mask(df[bool_col].isnull(), False)
        df[bool_col] = df[bool_col].mask(df[bool_col] == 'False', False)
        df[bool_col] = df[bool_col].mask(df[bool_col] == 'True', True)
        df[bool_col] = df[bool_col].astype(bool)

    # deal with other boolean columns that may require special handling
    for bool_col in ['collapsed', 'no_follow', 'is_submitter']:
        df[bool_col] = df[bool_col].mask(df[bool_col] == 'True', True)
        df[bool_col] = df[bool_col].mask(df[bool_col] != 'True', False)
        df[bool_col] = df[bool_col].astype(bool)

    # for select str columns, set NaN entries "unknown"
    for col in ['author_flair_text_color', 'author_flair_background_color',
                'distinguished', 'name', 'permalink']:
        df[col] = df[col].mask(df[col].isnull(), 'unknown')
        df[col] = df[col].astype(str)

    # for select str columns, set NaN entries to "none"
    for col in ['author_flair_css_class', 'author_flair_text']:
        df[col] = df[col].mask(df[col].isnull(), 'none')
        df[col] = df[col].astype(str)

    # for select str cols, set non-str entries (e.g, NaN, [], False) to 'not_applicable'
    str_cols = ['author_flair_type', 'collapsed_reason']
    for str_col in str_cols:
        df[str_col] = df[str_col].mask(df[str_col].isnull(), 'not_applicable')
        df[str_col] = df[str_col].mask(df[str_col] == '[]', 'not_applicable')
        df[str_col] = df[str_col].mask(df[str_col] == 'False', 'not_applicable')
        df[str_col] = df[str_col].mask(df[str_col].apply(type) != str, 'not_applicable')
        df[str_col] = df[str_col].astype(str)

    # for select int cols, convert NaN entries to 0
    for col in ['score', 'ups']:
        df[col] = df[col].mask(df[col].isnull(), 0)
        df[col] = df[col].astype(float).astype(int)  # just to be safe, lest decimals

    # set NaN entries in `controversiality` to 0, as this appears to be a binary column
    df.controversiality = df.controversiality.mask(df.controversiality.isnull(), 0)
    df.controversiality = df.controversiality.astype(float).astype(int)

    # set `downs` to 0, since this field has been deprecated by Reddit
    df.downs = 0
    df.downs = df.downs.astype(int)

    # convert non-numeric entries in `edited` to NaN
    df.edited = df.edited.mask(df.edited.isin(['True', True, 'False', False, '[]']), nan)
    df.edited = df.edited.astype(float)

    # set `subreddit_name_prefixed` to be "r/" + `subreddit`, as intended I think
    df.subreddit_name_prefixed = 'r/' + df.subreddit.astype(str)
    df.subreddit_name_prefixed = df.subreddit_name_prefixed.astype(str)

    # set `subreddit_type` to "public" globally, since only public subs can be scraped
    df.subreddit_type = 'public'
    df.subreddit_type = df.subreddit_type.astype(str)

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
                    'invalid samples from data set. Write cleaned results to a new '
                    'directory named "<input-directory>_clean".'
    )
    parser.add_argument(
        'input_directory',
        type=full_path,
        default=DEFAULTS['PATHS']['DIRS']['ALL_FIELDS'],
        help='path to directory with .parquet files to clean; files should contain '
             'all fields originally scraped from Reddit (default: %(default)s)'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_directory_basename = args.input_directory.name + '_clean'
    output_directory = args.input_directory.parent / output_directory_basename
    output_directory.mkdir(exist_ok=True, parents=True)

    df = load_raw_data(args.input_directory)
    df, nrows_original, nrows_final = drop_invalid_rows(df)

    ndropped = nrows_original - nrows_final
    ndropped_pct = ndropped / nrows_original
    print(f'dropped: {ndropped:,} rows ({ndropped_pct:.2%})')

    # ensure no columns contain mixed data types
    df = normalize_dtypes_by_column(df)

    df.to_parquet(output_directory, **DEFAULTS['IO']['TO_PARQUET_PARAMS'])


if __name__ == '__main__':
    main()
