"""
Object representing a dataset and all its partitions.
"""

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List, Optional, Union

from scipy.sparse.base import spmatrix
from sklearn.model_selection import train_test_split

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

        # load and preprocess comments (these attrs are dask.dataframes)
        self.raw_data = self.load_raw_data(config)
        self.preprocessed_text = self.preprocess(config, self.raw_data)

        categorical_labels, ids = dask.compute(self.raw_data.subreddit.astype(object).values,
                                               self.raw_data.index.values)

        # integer-encode labels and get class names (these attrs are numpy.ndarrays)
        self.categorical_labels = categorical_labels
        self.labels, class_names = pd.factorize(self.categorical_labels)
        self.class_name_mappings = dict(enumerate(class_names))
        self.map_to_categorical = np.vectorize(self.class_name_mappings.get)

        # set other misc attributes
        self.ids: np.ndarray = ids
        self.size: int = len(self)

        # extract features
        if config.features_file is None:
            self.features = self.extract_features(config, self.preprocessed_text)
        else:
            self.features = self.load_features(config)

        self.describe()
        self.partition(config)


    def __len__(self):
        return len(self.ids)


    def describe(self) -> None:     # TODO

        # `self.features`, `self.labels`, and `self.categorical_labels` will all be
        # defined before this is called

        # `self.features` will be a `np.ndarray` or a scipy sparse matrix, `self.labels`
        # will be a `np.ndarray`, and `self.categorical_labels` will be a `dask.Series`

        # in this method, just compute some summary statistics about distributions of
        # classes and such, whatever i want to know in order to understand my model's
        # performance better (see notes on phone for many ideas)

        # these stats can be referenced when creating the `Report` object


        self.metadata = Metadata(self)

        # compute differences between train and test metadata to evaluate how similarly
        # distributed the train and test sets are, both within and across subreddits
#        self.metadata.diffs = ...
        pass # TODO




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

        cls.extractor = extractor(**config.extractor_kwargs)
        return cls.extractor.fit_transform(preprocessed_text)


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

        indices = np.arange(self.size)
        train, test = train_test_split(indices, **config.train_test_split_kwargs)

        self.train = Partition(features=self.features[train],
                               labels=self.labels[train],
                               categorical_labels=self.map_to_categorical(self.labels[train]),
                               ids=self.ids[train],
                               class_name_mappings=self.class_name_mappings)

        self.test = Partition(features=self.features[test],
                              labels=self.labels[test],
                              categorical_labels=self.map_to_categorical(self.labels[test]),
                              ids=self.ids[test],
                              class_name_mappings=self.class_name_mappings)


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
                 categorical_labels: np.ndarray,
                 ids: np.ndarray,
                 class_name_mappings: Dict[int, str]):
        """
        Create specialized `Dataset` object for the train or test set partitions.

        Parameters
        ----------
        features : Union[np.ndarray, spmatrix]
            extracted features
        labels : np.ndarray
            binarized labels
        categorical_labels : np.ndarray
            subreddit names - useful for `describe`
        ids : np.ndarray
            comment IDs
        class_name_mappings ; Dict[int, str]
            mappings between integer labels and categorical labels
        """

        self.features = features
        self.labels = labels
        self.categorical_labels = categorical_labels
        self.ids = ids
        self.size = ids.size
        self.class_name_mappings = class_name_mappings

        self.describe()


class Metadata:
    """A container for housing all interesting `Dataset` metadata as attributes."""

    def __init__(self, dataset: Union[Dataset, Partition]) -> None:
        class_names = dataset.class_name_mappings.values()
        _, n_by_class = np.unique(dataset.labels, return_counts=True)
        self.class_distributions_n = pd.Series(n_by_class, index=class_names)
        self.class_distributions_pct = self.class_distributions_n / dataset.size

        self.feature_matrix = self.compute_feature_metadata_matrix(dataset)


    @staticmethod
    def compute_feature_metadata_matrix(
        dataset: Union[Dataset, Partition]
    ) -> pd.DataFrame:

        if isinstance(dataset.features, spmatrix):
            token_to_index = {i : t for  t, i in dataset.extractor.vocabulary_.items()}
            feature_matrix = pd.DataFrame.sparse.from_spmatrix(dataset.features,
                                                               index=dataset.ids)
            feature_matrix.columns = feature_matrix.columns.map(token_to_index)
#            ... but what can i do with this? with CountVectorizer anyway, the dims are enormous...

        elif isinstance(dataset.features, np.ndarray):
            err = ('feature metadata matrix computation for np.ndarray feature sets '
                   'not yet implemented')
            raise NotImplementedError(err) # TODO

        else:
            err = ('Currently feature metadata matrices can only be computed for '
                   f'features of type "{type(np.ndarray(0))}" or "{type(spmatrix)}", '
                   f'but the "{type(dataset.extractor)}" feature extractor produces '
                   f'type {type(dataset.features)}".')
            raise NotImplementedError(err) # keep this one

        return feature_matrix
