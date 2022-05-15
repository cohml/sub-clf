"""
Object representing a dataset and all its partitions.
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Optional, Union

from scipy.sparse.base import spmatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from sub_clf.experiment.available import AVAILABLE
from sub_clf.experiment.config import Config
from sub_clf.preprocess.base import MultiplePreprocessorPipeline
from sub_clf.util.defaults import DEFAULTS
from sub_clf.util.io import FEATURE_LOADERS, load_raw_data
from sub_clf.util.utils import pretty_dumps


class Dataset:
    """The total set of all comments used to train and evaluate a model."""

    def __init__(self, config: Config) -> None:
        """
        Read in data and return features.

        Parameters
        ----------
        config : Config
            an object enumerating all parameters for your experiment
        """

        self.raw_data = self.load_raw_data(config)
        self.preprocessed_text = self.preprocess(config, self.raw_data)
        self.labels = LabelBinarizer().fit_transform(self.raw_data.subreddit)
        self.categorical_labels = self.raw_data.subreddit
        self.ids = self.raw_data.index
        self.size = len(self)

        if config.features_file is not None:
            self.features = self.load_features(config)
        else:
            self.features = self.extract_features(config)

        self.describe()
        self.partition(config)


    def __len__(self):
        return len(self.ids)


    def describe(self) -> None:     # TODO

        # `self.features`, `self.labels`, and `self.categorical_labels` will all be
        # defined before this is called

        # `self.features` will be a `np.ndarray` or a scipy spare matrix, `self.labels`
        # will be a `np.ndarray`, and `self.categorical_labels` will be a `dask.Series`

        # in this method, just compute some summary statistics about distributions of
        # classes and such, whatever i want to know in order to understand my model's
        # performance better (see notes on phone for many ideas)

        # these stats can be referenced when creating the `Report` object

        pass


    @classmethod
    def extract_features(cls,
                         config: Config,
                         preprocessed_text) -> None:
        """Extract features as specified in the config."""

        extractors = AVAILABLE['FEATURE_EXTRACTORS']
        extractor = extractors.get(config.extractor)

        if extractor is None:
            err = (f'"{config.extractor}" is not a recognized feature extractor. '
                   f'Please select one from the following: {pretty_dumps(extractors)}')
            raise KeyError(err)

        extractor = extractor(**config.extractor_kwargs)
        return extractor.fit_transform(preprocessed_text)


    def load_features(self, config: Config) -> None:
        """Read in feature values from the specified file."""

        # see imported `FEATURE_LOADERS` object for bespoke feature-loading functions,
        # which vary by feature extractor

        err = 'Loading features from a file is not yet implemented.'
        raise NotImplementedError(err)

        return None


    @classmethod
    def load_raw_data(cls, config: Config) -> None:
        """Read and merge all raw data in the specified directory/files."""

        return load_raw_data(config.raw_data_directory or config.raw_data_filepaths)


    def partition(self, config: Config) -> None:
        """Split data into train and test sets."""

        subs = self.raw_data.subreddit
        ids = self.raw_data.index.compute()
        indices = np.arange(self.size)

        (train_ids, test_ids,
         train_idx, test_idx) = train_test_split(ids, indices,
                                                 **config.train_test_split_kwargs)

        self.train = Partition(features=self.features[train_idx],
                               labels=self.labels[train_idx],
                               categorical_labels=subs.loc[subs.index.isin(train_ids)],
                               ids=train_ids)

        self.test = Partition(features=self.features[test_idx],
                              labels=self.labels[test_idx],
                              categorical_labels=subs.loc[subs.index.isin(test_ids)],
                              ids=test_ids)


    @classmethod
    def preprocess(cls, config: Config, raw_data: dd.DataFrame) -> None:
        """Apply preprocessing steps as specified in the config."""

        if 'preprocessing_pipeline' in config:

            pipelines = AVAILABLE['PREPROCESSING']['PIPELINES']
            (pipeline_name, pipeline_verbosity), = config.preprocessing_pipeline.items()
            pipeline = pipelines.get(pipeline_name)

            if pipeline is None:
                err = (f'"{pipeline}" is not a recognized preprocessing pipeline. '
                       f'Please select one from the following: {pretty_dumps(pipelines)}')
                raise KeyError(err)

            pipeline = pipeline(**pipeline_verbosity)

        else:

            preprocessors = AVAILABLE['PREPROCESSING']['PREPROCESSORS']
            initialized_preprocessors = []

            for preprocessor in config.preprocessors:
                (preprocessor_name, preprocessor_kwargs), = preprocessor.items()
                preprocessor = preprocessors.get(preprocessor_name)

                if preprocessor is None:
                    err = (f'"{preprocessor_name}" is not a recognized preprocessor. '
                           f'Please select from the following: {pretty_dumps(preprocessors)}')
                    raise KeyError(err)

                preprocessor = preprocessor(**preprocessor_kwargs)
                initialized_preprocessors.append(preprocessor)

            pipeline = MultiplePreprocessorPipeline(*initialized_preprocessors)

        return pipeline.preprocess(raw_data.text)


class Partition(Dataset):
    """A subset of the total dataset, for example a train or test set."""

    def __init__(self,
                 features: Union[np.ndarray, spmatrix],
                 labels: np.ndarray,
                 categorical_labels: dd.Series,
                 ids: pd.Series):
        """
        Create specialized `Dataset` object for the train or test set partitions.

        Parameters
        ----------
        features : Union[np.ndarray, spmatrix]
            extracted features
        labels : np.ndarray
            binarized labels
        categorical_labels : dd.Series
            subreddit names - useful for `describe`
        ids : pd.Series
            comment IDs
        """

        ids = dd.from_pandas(ids.to_series(), npartitions=DEFAULTS['NCPU'])

        self.features = features
        self.labels = labels
        self.categorical_labels = categorical_labels
        self.ids = ids
        self.size = ids.size

        self.describe()
