"""
Default value constants for importing into other scripts.
"""


from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULTS = {
    'LOG' : {
        'FORMAT' : '%(asctime)s : %(levelname)s : %(module)s:%(funcName)s:%(lineno)d : %(message)s',
    },
    'PATHS' : {
        'FILES' : {
            'LOG_CONFIG' : PROJECT_DIR / 'meta' / 'logging.cfg',
            'MY_SUBREDDITS_FILE' : PROJECT_DIR / 'meta' / 'my_subreddits.lst',
            'REDDIT_OAUTH_CREDENTIALS' : PROJECT_DIR / 'meta' / 'credentials.json'
        },
        'DIRS' : {
            'ALL_FIELDS' : PROJECT_DIR / 'data' / 'all_fields',
            'BODY_ONLY' : PROJECT_DIR / 'data' / 'body_only',
        }
    }
}
