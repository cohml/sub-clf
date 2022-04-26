"""
Use PRAW to pull comments from top and hot posts in one or more specified
subreddits and write the results to a series of CSVs, one per subreddit.
"""

import argparse
import logging
import logging.config
import pandas as pd
import praw

from datetime import datetime
from pathlib import Path
from prawcore.exceptions import Forbidden
from time import sleep
from typing import Iterable, List, Optional

from const import DEFAULTS
from login import connect


# set up logging
logging.config.fileConfig(DEFAULTS['PATHS']['FILES']['LOG_CONFIG'])
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(module)s:%(funcName)s:%(lineno)d : %(message)s')


def get_subreddits(subreddits_filepath: Path) -> List[str]:
    """
    Read list of subreddits to scrape comment data from.

    Parameters
    ----------
    subreddits_filepath : Path
        path to file enumerating subreddits to scrape comment data from

    Returns
    -------
    subreddits : List[str]
        names of subreddits to scrape comment data from
    """

    return subreddits_filepath.read_text().splitlines()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description='Scrape comments on top and hot Reddit posts in passed subreddits.'
    )
    subreddits = parser.add_mutually_exclusive_group()
    subreddits.add_argument(
        '-s', '--subreddits',
        nargs='+',
        help='whitespace-delimited list of subreddits to scrape; passed names are '
             'case-sensitive and must exactly match how they appear on Reddit'
    )
    subreddits.add_argument(
        '-f', '--subreddits-filepath',
        type=lambda s: Path(s).resolve(),
        default=DEFAULTS['PATHS']['FILES']['MY_SUBS_FILE'],
        help='path to file enumerating subreddits to scrape comment data from '
             '(default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--output-directory',
        type=lambda s: Path(s).resolve(),
        default=DEFAULTS['PATHS']['DIRS']['ALL_FIELDS'],
        help='path to directory to write output files to (one per subreddit)'
             '(default: %(default)s)'
    )
    parser.add_argument(
        '-l', '--log-filepath',
        type=lambda s: Path(s).resolve(),
        help='path to log file; if not passed, no logging occurs'
    )
    parser.add_argument(
        '-p', '--sleep-duration',
        type=int,
        default=5,
        help='number of minutes to sleep between scraping subs (default: '
             '%(default)s)'
    )
    parser.add_argument(
        '-r', '--resume',
        action='store_true',
        help='skip subreddits which have already been scraped (pass this if '
             'the previous attempt to scrape terminated early, e.g. '
             'RecursionError) (default: %(default)s)'
    )
    return parser.parse_args()


def scrape_posts(reddit: praw.Reddit,
                 subreddit: str,
                 subreddit_counter: str
                 ) -> Optional[List[praw.reddit.Submission]]:
    """
    Scrape up to 1000 top + hot posts from each of the passed subreddits.

    Parameters
    ----------
    reddit : praw.Reddit
        a Reddit instance
    subreddit : str
        display name of a single subreddit
    subreddit_counter : str
        string representation of number of subreddits scraped so far

    Returns
    -------
    posts : Optional[List[praw.reddit.Submission]]
        scraped posts; will be `NoneType` if subreddit cannot be scraped (e.g.,
        subreddit is private, or the name is misspelled)
    """

    logger.info(f'Scraping subreddit "{subreddit}" ({subreddit_counter})')

    try:
        top_posts = reddit.subreddit(subreddit).top(limit=None)
        hot_posts = reddit.subreddit(subreddit).hot(limit=None)
    except Forbidden:
        logger.error(f'ForbiddenError triggered by subreddit "{subreddit}" '
                     '-- check if private, or typo in name '
                     '-- skipping subreddit')
        return None

    posts = list(top_posts) + list(hot_posts)
    logger.info(f'Scraping subreddit "{subreddit}" ({subreddit_counter}) '
                f'-- {len(posts)} posts')

    return posts


def traverse_comment_threads(posts: List[praw.reddit.Submission],
                             subreddit_counter: str
                             ) -> List[pd.Series]:
    """
    Scrape all comments across all posts scraped from a single subreddit.

    Parameters
    ----------
    posts : List[praw.reddit.Submission]
        all posts scraped from a single subreddit
    subreddit_counter : str
        string representation of number of subreddits scraped so far

    Returns
    -------
    comments : List[pd.Series]
        all scraped comments from a single subreddit
    """

    def traverse(comment: praw.reddit.Comment) -> Iterable[pd.Series]:
        """Recursivly traverse single-comment tree of arbitrary depth."""
        if isinstance(comment, praw.models.MoreComments):
            for comment in comment.comments():
                yield from traverse(comment)

        else:
            nonlocal subreddit_counter, post_counter, num_comment
            num_comment += 1
            logger.info(f'Scraping subreddit "{post.subreddit.display_name}" ({subreddit_counter}) '
                        f'-- post "{post.id}" ({post_counter}) '
                        f'-- comment "{comment.id}" ({num_comment}/{post.num_comments})')
            yield pd.Series(vars(comment))

            try:
                for reply in comment.replies:
                    yield from traverse(reply)
            except RecursionError:  # if comment tree is deeper than Python recursion limit
                logger.warning(f'RecursionError triggered by post "{post.id}", comment "{comment.id}" '
                               '-- skipping children')
                pass

    comments = []
    num_posts = len(posts)
    for num_post, post in enumerate(posts, start=1):
        post_counter = f'{num_post}/{num_posts}'
        logger.info(f'Scraping subreddit "{post.subreddit.display_name}" ({subreddit_counter}) '
                    f'-- post "{post.id}" ({post_counter}, {post.url[8:]}) '
                    f'-- {post.num_comments} comments')

        num_comment = 0
        for comment in post.comments:
            comment_thread = traverse(comment)
            comments.extend(comment_thread)

    return comments


def write_to_csv(comments: pd.DataFrame,
                 subreddit: str,
                 output_directory: Path
                 ) -> None:
    """
    Write comments from a single subreddit to CSV in specified directory.
    If output filename exists, meaning comments for the passed subreddit
    have already been scraped and saved, then merge and overwrite.

    Parameters
    ----------
    comments : pd.DataFrame
        all scraped comments from a single subreddit
    subreddit : str
        display name of a single subreddit
    output_directory : Path
        path to directory to write output files to
    """

    output_file = output_directory / (subreddit + '.csv')

    if output_file.exists():
        logger.info(f'Merging subreddit "{subreddit}" '
                    f'-- new data with existing {output_file}')
        comments = pd.concat([comments, pd.read_csv(output_file)])

    # drop useless columns containing praw technical metadata, and drop comments
    # scraped multiple times (which could happen if script failed previously, or
    # occasionally for a post tagged as both "top" and "new"), then write to CSV
    logger.info(f'Writing {output_file}')
    praw_junk = {'regex' : r'^_'}
    dropcols = comments.filter(**praw_junk).columns
    (comments.drop(columns=dropcols)
             .drop_duplicates(subset='id')
             .to_csv(output_file, index=False))


def main() -> None:
    # parse command line arguments
    args = parse_args()

    # make output directory if needed
    args.output_directory.mkdir(exist_ok=True, parents=True)

    # optionally log to file, appending if already exists
    if args.log_filepath:
        args.log_filepath.parent.mkdir(exist_ok=True, parents=True)
        mode = 'a' if args.log_filepath.exists() else 'w'
        log_filepath_handler = logging.FileHandler(args.log_filepath,
                                                   encoding='utf-8',
                                                   mode=mode)
        log_filepath_handler.setFormatter(formatter)
        logger.addHandler(log_filepath_handler)

    # connect to reddit
    reddit = connect()

    # get list of subs to scrape data from
    if args.subreddits:
        subreddits = args.subreddits
    else:
        subreddits = get_subreddits(args.subreddits_filepath)

    # optionally skip subreddits from which comments have already been scraped
    if args.resume:
        already_scraped = [p.stem for p in args.output_directory.glob('*.csv')]
        for subreddit in sorted(already_scraped, key=str.lower):
            if subreddit in subreddits:
                logger.info(f'Skipping subreddit "{subreddit}"')
                subreddits.remove(subreddit)

    num_subreddits = len(subreddits)

    for num_subreddit, subreddit in enumerate(subreddits, start=1):
        subreddit_counter = f'{num_subreddit}/{num_subreddits}'

        # scrape posts
        posts = scrape_posts(reddit, subreddit, subreddit_counter)

        # skip subreddit if error accessing its contents
        if posts is None:
            continue

        # scrape all comments across all scraped posts
        comments = traverse_comment_threads(posts, subreddit_counter)
        comments = pd.DataFrame(comments)

        # write to output
        write_to_csv(comments, subreddit, args.output_directory)

        if args.sleep_duration > 0 and num_subreddit < num_subreddits:
            logger.info(f'Sleeping {args.sleep_duration} minutes ...')
            sleep(60 * args.sleep_duration)


if __name__ == '__main__':
    main()
