"""
Default parameter values and other constants for importing into other scripts.
"""


from multiprocessing import cpu_count
from pathlib import Path


_PARQUET_SCHEMA = {
    'comment_id' : 'str',
    'post_id' : 'str',
    'subreddit' : 'str',
    'text' : 'str',
    'upvotes' : 'int64'
}

_PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULTS = {
    'CONFIG' : {
        'extractor_kwargs' : {},
        'features_file' : None,
        'model_kwargs' : {},
        'overwrite_existing' : False,
        'raw_data_directory' : None,
        'raw_data_filepaths' : None,
        'save_features': False,
        'save_metadata': False,
        'save_model': False,
        'save_preprocessed_texts': False,
        'save_train_test_ids': False,
        'train_test_split_kwargs': {}
    },
    'IO' : {
        'READ_PARQUET_KWARGS' : {
            'blocksize' : 1e8,
            'engine' : 'pyarrow',
            'index' : 'comment_id',
        },
        'TO_PARQUET_KWARGS' : {
            'compression' : 'gzip',
            'name_function' : lambda i: f'{i+1:05}.parquet.gz',
            'partition_on' : ['subreddit', 'post_id'],
            'schema' : _PARQUET_SCHEMA
        },
    },
    'LOG' : {
        'FORMAT' : '%(asctime)s : %(levelname)s : %(module)s:%(funcName)s:%(lineno)d : %(message)s',
    },
    'NCPU' : cpu_count(),
    'PATHS' : {
        'FILES' : {
            'LOG_CONFIG' : _PROJECT_DIR / 'meta' / 'logging.cfg',
            'MY_SUBREDDITS_FILE' : _PROJECT_DIR / 'meta' / 'my_subreddits.lst',
            'REDDIT_OAUTH_CREDENTIALS' : _PROJECT_DIR / 'meta' / 'credentials.json'
        },
        'DIRS' : {
            'ALL_FIELDS' : _PROJECT_DIR / 'data' / 'all_fields',
        }
    }
}
