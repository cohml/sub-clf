"""
Default value constants for importing into other scripts.
"""


from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULTS = {
    'DTYPES' : {
        'all_awardings': 'object',
         'approved_at_utc': 'float64',
         'approved_by': 'float64',
         'archived': 'object',
         'associated_award': 'float64',
         'author': 'object',
         'author_cakeday': 'float64',
         'author_flair_background_color': 'object',
         'author_flair_css_class': 'object',
         'author_flair_richtext': 'float64',
         'author_flair_template_id': 'float64',
         'author_flair_text': 'object',
         'author_flair_text_color': 'object',
         'author_flair_type': 'object',
         'author_fullname': 'float64',
         'author_is_blocked': 'float64',
         'author_patreon_flair': 'float64',
         'author_premium': 'object',
         'awarders': 'float64',
         'banned_at_utc': 'float64',
         'banned_by': 'float64',
         'body': 'object',
         'body_html': 'float64',
         'can_gild': 'object',
         'can_mod_post': 'float64',
         'collapsed': 'bool',
         'collapsed_because_crowd_control': 'float64',
         'collapsed_reason': 'object',
         'collapsed_reason_code': 'float64',
         'comment_type': 'float64',
         'controversiality': 'int64',
         'created': 'object',
         'created_utc': 'object',
         'depth': 'object',
         'distinguished': 'object',
         'downs': 'int64',
         'editable': 'float64',
         'edited': 'float64',
         'gilded': 'int64',
         'gildings': 'object',
         'id': 'object',
         'is_submitter': 'bool',
         'likes': 'float64',
         'link_id': 'object',
         'locked': 'object',
         'mod_note': 'float64',
         'mod_reason_by': 'float64',
         'mod_reason_title': 'object',
         'mod_reports': 'float64',
         'name': 'object',
         'no_follow': 'bool',
         'num_reports': 'float64',
         'parent_id': 'object',
         'permalink': 'object',
         'removal_reason': 'float64',
         'report_reasons': 'float64',
         'saved': 'float64',
         'score': 'int64',
         'score_hidden': 'object',
         'send_replies': 'object',
         'stickied': 'object',
         'subreddit': 'object',
         'subreddit_id': 'object',
         'subreddit_name_prefixed': 'object',
         'subreddit_type': 'object',
         'top_awarded_type': 'float64',
         'total_awards_received': 'int64',
         'treatment_tags': 'object',
         'unrepliable_reason': 'float64',
         'ups': 'int64',
         'user_reports': 'float64'
    },
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
