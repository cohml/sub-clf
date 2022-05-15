"""
Logic for running a machine learning experiment end to end.
"""


import argparse
import dask.dataframe as dd
import sys

from pathlib import Path
from typing import Optional

from sub_clf.experiment.config import Config
from sub_clf.experiment.dataset import Dataset
from sub_clf.experiment.experiment import Experiment
from sub_clf.util.utils import full_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a single machine learning experiment end to end using '
                    'parameters enumerated in a config file.'
    )
    parser.add_argument(
        'config_filepath',
        type=full_path,
        help='Path to a config .yaml enumerating parameters for your experiment.'
    )
    return parser.parse_args()


def run(args: Optional[argparse.Namespace] = None) -> None:
    if args:
        config_filepath = args.config_filepath
    else:
        config_filepath = Path(sys.argv[1])

    config = Config(config_filepath)
    dataset = Dataset(config)
    experiment = Experiment(config, dataset)

    experiment.run()


if __name__ == '__main__':
    args = parse_args()
    run(args)
