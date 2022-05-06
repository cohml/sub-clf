"""
Mappings between the available feature extractors and the
functions needed to save/load their outputs to/from files.
"""


import dask.dataframe as dd

from functools import partial
from numpy import load, savez_compressed
from pathlib import Path
from scipy.sparse import load_npz, save_npz
from typing import List, Union

from util.defaults import DEFAULTS


def load_raw_data(parent_directory: Union[List[Path], Path],
                  subreddit: str = '*',
                  **kwargs
                 ) -> dd.DataFrame:
    """Read and concatenate .parquet across multiple subreddits."""

    _read_parquet = partial(dd.read_parquet, **DEFAULTS['IO']['READ_PARQUET_KWARGS'])

    if isinstance(parent_directory, Path):
        subreddit_directories = parent_directory.glob(f'subreddit={subreddit}')
    else:
        subreddit_directories = parent_directory

    parquets = map(_read_parquet, subreddit_directories)
    return dd.concat(list(parquets))


FEATURE_LOADERS = {
    'HashingVectorizer' : load_npz,
    'CountVectorizer' : load_npz,
    'LgEmbeddingsVectorizer' : load,
    'TfidfTransformer' : load_npz,
    'TfidfVectorizer' : load_npz,
    'TrfEmbeddingsVectorizer' : load
}

FEATURE_SAVERS = {
    'HashingVectorizer' : save_npz,
    'CountVectorizer' : save_npz,
    'LgEmbeddingsVectorizer' : savez_compressed,
    'TfidfTransformer' : save_npz,
    'TfidfVectorizer' : save_npz,
    'TrfEmbeddingsVectorizer' : savez_compressed
}
