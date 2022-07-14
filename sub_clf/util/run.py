"""
Logic for running a machine learning experiment end to end.
"""


import argparse
import dask.dataframe as dd

from pathlib import Path

from sub_clf.experiment.config import Config
from sub_clf.experiment.dataset import Dataset
from sub_clf.experiment.experiment import Experiment
from sub_clf.util.utils import full_path


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Apply parameters enumerated in a config file to either '
                    'preprocess raw data, extract features from preprocessed data, or '
                    'train a model on extracted features, depending on the command '
                    'used (i.e., "preprocess", "extract", or "train").'
    )
    parser.add_argument(
        'config_filepath',
        type=full_path,
        help='Path to a .yaml config file.'
    )

    args = parser.parse_args()
    operation = parser.prog  # will be either "preprocess", "extract", or "train"

    return args, operation


def run() -> None:
    args, operation = _parse_cli()

    config = Config(args.config_filepath, operation)
    dataset = Dataset(config)

    if operation == 'train':
        experiment = Experiment(config, dataset)
        experiment.run()
