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
    'CONFIG' : {    # optional fields only, mostly
        'preprocess' : {
            'overwrite_existing' : False,
            'raw_data_directory' : None,
            'raw_data_filepaths' : None,
            'resume' : True
        },
        'extract' : {
            'extractor_kwargs' : {},
            'overwrite_existing' : False,
            'preprocessed_data_directory' : None,
            'preprocessed_data_filepaths' : None,
            'scaler_pipeline' : None,
            'train_test_split_kwargs' : {}
        },
        'train' : {
            'model_kwargs' : {},
            'overwrite_existing' : False,
            'save_model' : False,
            'save_test_predictions' : False
        }
    },
    'IO' : {
        'READ_PARQUET_KWARGS' : {
            'blocksize' : 1e8,
            'engine' : 'pyarrow',
            'index' : 'comment_id',
            'parquet_file_extension' : (
                    '.parquet',
                    '.parquet.gz'
            ),
        },
        'TO_PARQUET_KWARGS' : {
            'compression' : 'gzip',
            'name_function' : lambda i: f'{i+1:06}.parquet.gz',
            'partition_on' : ['subreddit', 'post_id'],
            'schema' : _PARQUET_SCHEMA,
            'write_metadata_file' : False
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
            'RAW_DATA' : _PROJECT_DIR / 'data' / 'raw',
        }
    }
}
