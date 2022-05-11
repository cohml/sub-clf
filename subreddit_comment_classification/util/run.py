"""
Logic for running a machine learning experiment end to end.
"""


import argparse
import dask.dataframe as dd

from pathlib import Path

from ..experiment.config import Config
from ..experiment.dataset import Dataset
from ..experiment.experiment import Experiment
from ..util.misc import full_path


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


def run(config_filepath: Path) -> None:
    args = parse_args()

    config = Config(args.config_filepath)
    dataset = Dataset(config)
    experiment = Experiment(config, dataset)

    experiment.run()


if __name__ == '__main__':
    run()
