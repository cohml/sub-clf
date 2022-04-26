"""
To connect to Reddit, just import this module and run the `connect` function, or
just import and run the `connect` function directly.
"""


import json
import praw

from pathlib import Path


CREDENTIALS = Path(__file__).resolve().parent.parent / 'meta' / 'credentials.json'


def get_credentials(CREDENTIALS):
    """Load credentials from config file for OAuthorization"""

    with open(CREDENTIALS) as credentials:
        return json.load(credentials)


def connect():
    """Connect to Reddit API"""

    credentials = get_credentials(CREDENTIALS)
    return praw.Reddit(**credentials)
