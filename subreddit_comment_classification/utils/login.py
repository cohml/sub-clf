"""
Utility functions for easily connecting to the Reddit API.
"""


import json
import praw

from ..utils.defaults import DEFAULTS


CREDENTIALS_FILEPATH = DEFAULTS['PATHS']['FILES']['REDDIT_OAUTH_CREDENTIALS']


class Reddit:

    def __init__(self):
        """Log into Reddit's backend API."""
        with open(CREDENTIALS_FILEPATH) as credentials_fh:
            self.credentials = json.load(credentials_fh)
            self.session = praw.Reddit(**self.credentials)
