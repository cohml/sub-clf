"""
Utility functions for easily connecting to the Reddit API.
"""


import json
import praw

from pathlib import Path

from subreddit_comment_classification.src.utils.const import DEFAULTS


CREDENTIALS = DEFAULTS['PATHS']['FILES']['REDDIT_OAUTH_CREDENTIALS']


def get_credentials(CREDENTIALS):
    """Load credentials from config file for OAuthorization"""

    with open(CREDENTIALS) as credentials:
        return json.load(credentials)


def connect():
    """Connect to Reddit API"""

    credentials = get_credentials(CREDENTIALS)
    return praw.Reddit(**credentials)
