"""
Mappings between the available feature extractors and the
functions needed to save/load their outputs to/from files.
"""

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
