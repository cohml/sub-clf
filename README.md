# Project overview

This repository contains resources for training machine learning classifiers to predict
the subreddit from which an arbitrary comment most likely originated.

# Conda environment

Use of this repository requires a conda environment. To create one and install the
project's package into it:

    $ bash ./create_env.sh

> **_NOTE:_** `create_env.sh` can be run from anywhere, and will always install the
prefixed environment `./env` into the repository root.

# Data collection

To scrape new data, first ensure that a file enumerating your Reddit OAuth credentials
exists at `./subreddit_comment_classification/meta/credentials.json`. The file should
define the following fields:

```json
{
    "client_id": "...",
    "client_secret": "...",
    "user_agent": "..."
}
```

> **_NOTE:_** See the [PRAW documentation](https://praw.readthedocs.io/en/stable/getting_started/authentication.html)
for instructions on how to complete these fields.

Once your `credentials.json` is in place, to scrape new data from scratch using default
parameters:

    $ conda activate ./env
    $ scrape

To see argumements useful for customing your data collection away from the default
parameters:

    $ scrape --help

> **_NOTE:_** The scraped comments set will not be identical to the original data set.
This is because new "top" and "hot" posts are always being added, while the number you
can scrape from each category is capped at 1000. Regardless, the characteristics of all
newly scraped comments should approximate the original data set, given the sheer volume
of comments `scrape` will retrieve.

After scraping, some rows may contain data that is corrupt or otherwise invalid. To
identify and drop all rows thus affected:

    $ conda activate ./env
    $ clean
