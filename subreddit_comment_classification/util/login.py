"""
Object for navigating Reddit's back end API.
"""


import json
import praw

from ..util.defaults import DEFAULTS


CREDENTIALS_FILEPATH = DEFAULTS['PATHS']['FILES']['REDDIT_OAUTH_CREDENTIALS']


class Reddit:

    def __init__(self):
        credentials = CREDENTIALS_FILEPATH.read_text()
        self.credentials = json.loads(credentials)


    def login(self):
        self.session = praw.Reddit(**self.credentials)
