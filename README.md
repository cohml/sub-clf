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
exists at `./sub_clf/meta/credentials.json`. The file must define the following fields:

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

# Data preprocessing

To preprocess your scraped data in preparation for feature extraction, first create a
.yaml configuration file. Use the following example as a template:

    ./sub_clf/configs/configs/config_template_PREPROCESS.yaml

Once created, run the `preprocess` command and pass the path to your configuration file:

    $ preprocess <path/to/config>

With many millions of records, preprocessing can incur extremely long runtimes. Should
a job fail prematurely, you can effectively restart the preprocessing from where the
job failed by adding the following field to your configuration file:

    resume: true

Then rerun the above command.

# Feature extraction

To extract features from preprocessed data prior to training a model, first create a
.yaml configuration file. Use the following example as a template:

    ./sub_clf/configs/configs/config_template_EXTRACT.yaml

Once created, run the `extract` command and pass the path to your configuration file:

    $ extract <path/to/config>

# Model building

Currently under development...
