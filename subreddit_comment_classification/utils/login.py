"""
Object representing a Reddit session.
"""


import json
import praw

from ..utils.defaults import DEFAULTS


CREDENTIALS_FILEPATH = DEFAULTS['PATHS']['FILES']['REDDIT_OAUTH_CREDENTIALS']


class Reddit:

    def __init__(self):
        """Log into Reddit's backend API."""

        credentials = CREDENTIALS_FILEPATH.read_text()
        self.credentials = json.loads(credentials)
        self.session = praw.Reddit(**self.credentials)
