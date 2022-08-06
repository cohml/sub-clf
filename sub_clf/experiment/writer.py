"""
Object designed to write all output files across all `preprocess`, `extract`, and
`train` operations. In the case of `preprocess` operations, for technical reasons,
this object is designed for use as the final transformation step of an sklearn
preprocessing pipeline.
"""


import dask.dataframe as dd
import json
import numpy as np
import pandas as pd
import yaml

from copy import deepcopy
from overrides import overrides
from typing import Dict, Union

from scipy.sparse import spmatrix

from sub_clf.experiment.config import Config
from sub_clf.preprocess.base import SinglePreprocessor
from sub_clf.util.defaults import DEFAULTS
from sub_clf.util.io import FEATURE_SAVERS


class OutputWriter(SinglePreprocessor):

    @overrides
    def __init__(self, config: Config):
        name = config.operation.capitalize() + 'Writer'
        super().__init__(name=name)

        self.force = {'exist_ok' : config.overwrite_existing, 'parents' : True}
        self.config = config


    def transform(self, data: pd.DataFrame) -> None:
        self.write_preprocessed_data(data)


    def write_config(self) -> None:
        """
        Save config parameters in .json and .yaml formats.

        Of all outputs across all `preprocess`, `extract`, and `train` operations,
        this file is always written first.
        """

        config_output_directory = self.config.output_directory / 'configs'
        config_output_directory.mkdir(**self.force)
        config_output_filepath_stem = config_output_directory / f'config'

        # clean up config fields for immediate reusability and JSON serializability
        serializable = deepcopy(vars(self.config))
        for key, value in deepcopy(serializable).items():
            if key == 'dict' or value is None:
                serializable.pop(key)
            elif key.endswith('directory'):
                serializable[key] = str(value)
            elif key.endswith('filepaths'):
                serializable[key] = [str(filepath) for filepath in value]

        config_savers = {'.json' : json.dump,
                         '.yaml' : yaml.safe_dump}

        for suffix, config_saver in config_savers.items():
            with config_output_filepath_stem.with_suffix(suffix).open('w') as fh:
                config_saver(serializable, fh, indent=4, sort_keys=True)


    def write_features(self, features: Dict[str, Union[np.ndarray, spmatrix]]) -> None:
        """Save train and test set features to .npz files."""

        features_output_directory = self.config.output_directory / 'features'
        features_output_directory.mkdir(**self.force)

        feature_saver = FEATURE_SAVERS[self.config.extractor]

        for partition_name, partition_features in features.items():
            feature_saver(
                features_output_directory / f'{partition_name}.npz',
                partition_features
            )


    def write_comment_ids(self, partitions: Dict[str, dd.core.DataFrame]) -> None:
        """Save unique comment IDs for the train and test set partitions."""

        comment_ids_output_directory = self.config.output_directory / 'ids'
        comment_ids_output_directory.mkdir(**self.force)

        to_parquet_kwargs = DEFAULTS['IO']['TO_PARQUET_KWARGS'].copy()
        for irrelevant_kwarg in ['partition_on', 'schema']:
            to_parquet_kwargs.pop(irrelevant_kwarg)

        for partition_name, partition_data in partitions.items():
            partition_data.index.to_frame(name='').to_parquet(
                comment_ids_output_directory / partition_name,
                **to_parquet_kwargs
            )


    def write_preprocessed_data(self, data: pd.DataFrame) -> None:
        """
        Write the preprocessed data for a single subreddit to a series of parquet.gz
        files, and append the associated comment IDs to a .tmp file (for potential use
        with `resume`).

        This method is only ever called at the end of an sklearn `Pipeline` subclass
        applied to a dask `DataFrameGroupBy` object.
        """

        preprocessed_data_output_kwargs = {
            'path' : self.config.output_directory / 'data',
            **DEFAULTS['IO']['TO_PARQUET_KWARGS']
        }

        (dd.from_pandas(data, npartitions=DEFAULTS['NCPU'])
           .to_parquet(**preprocessed_data_output_kwargs))

        ids_tmp_file = self.config.output_directory / 'data' / 'comment_ids.tmp'
        with ids_tmp_file.open('a') as f:
            for comment_id in data.index:
                print(comment_id, file=f)
