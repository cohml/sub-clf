"""
Mappings between the available feature extractors and the
functions needed to save/load their outputs to/from files.
"""


import dask.dataframe as dd

from functools import partial
from numpy import load, savez_compressed
from pathlib import Path
from scipy.sparse import load_npz, save_npz


_read_parquet = partial(dd.read_parquet, blocksize=1e8, dtype=object, engine='pyarrow')


def read_parquets(parent_directory: Path, subreddit: str = '*') -> dd.DataFrame:
    """Read and concatenate .parquet across multiple subreddits."""

    subreddit_directories = parent_directory.glob(f'subreddit={subreddit}')
    parquets = map(_read_parquet, subreddit_directories)
    return dd.concat(list(parquets), ignore_index=True)


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

RAW_DATA_LOADERS = {
    'csv' : partial(dd.read_csv,
                    dtype=object,
                    blocksize=1e8,
                    usecols=['body', 'subreddit']),
    'parquet' : partial(_read_parquet,
                        columns=['body', 'subreddit'])
}
