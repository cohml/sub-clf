"""
Default value constants for importing into other scripts.
"""


from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULTS = {
    'LOG' : {
        'LEVEL' : 'INFO',
        'FMT' : '{}  **{:^10}**  {}',
    },
    'PATHS' : {
        'FILES' : {
            'MY_SUBS_FILE' : PROJECT_DIR / 'meta' / 'my_subreddits.lst',
            'LOG_CONFIG' : PROJECT_DIR / 'meta' / 'logging.cfg',
        },
        'DIRS' : {
            'ALL_FIELDS' : PROJECT_DIR / 'data' / 'all_fields',
            'BODY_ONLY' : PROJECT_DIR / 'data' / 'body_only',
        }
    }
}
