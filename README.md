# Conda environment

Use of this repository requires a conda environment. To create one and install the
project's package into it:

    $ bash ./create_env.sh

> **_Note_:** `create_env.sh` can be run from anywhere, and will always install the
prefixed environment `./env` into the repository root.

# Data collection

To scrape new data from scratch using default parameters:

    $ conda activate ./env
    $ scrape

To see argumements useful for customing your data collection away from the default
parameters:

    $ scrape --help

> **_Note:_** The scraped comments set will not be identical to the original data set.
This is because new "top" and "hot" posts are always being added, while the number you
can scrape from each category is capped at 1000. Regardless, the characteristics of all
newly scraped comments should approximate the original data set, given the sheer volume
of comments `scrape` will retrieve.
