"""
Extract features from comment body raw texts and save to CSV, including column
for class labels.
"""


import argparse
import pandas as pd
# import spacy

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


def get_lexicon(comments: pd.DataFrame) -> pd.DataFrame:
    """
    Construct list of all unique tokens across all comments.
    """
    pass


def normalize_body(comments: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize comment body raw texts into a standardized format.
    """
    pass


def parse_args() -> None:
    """
    Parse command line arguments.
    """
    pass


def read_comment(body_only_directory: Path) -> pd.DataFrame:
    """
    Read all CSVs containing comment body raw texts and concatenate into a
    single df for efficient preprocessing and feature extraction.
    """
    pass


def main() -> None:
    args = parse_args()

    comments = read_comments(args.body_only_directory)
    comments = normalize_body(comments)
    lexicon = get_lexicon(comments)


if __name__ == '__main__':
    main()
