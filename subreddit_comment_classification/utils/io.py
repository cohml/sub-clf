"""
Mappings between the available feature extractors and the
functions needed to save/load their outputs to/from files.
"""


from dask.dataframe import read_csv, read_parquet
from functools import partial
from numpy import load, savez_compressed
from scipy.sparse import load_npz, save_npz


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
    'csv' : partial(read_csv,
                    dtype=object,
                    blocksize=1e8),
    'parquet' : partial(read_parquet,
                        dtype=object,
                        blocksize=1e8,
                        engine='fastparquet')
}
