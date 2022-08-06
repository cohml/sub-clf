"""
Object representing a dataset and all its partitions.
"""

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from scipy.sparse import spmatrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sub_clf.experiment.available import AVAILABLE
from sub_clf.experiment.config import Config
from sub_clf.experiment.writer import OutputWriter
from sub_clf.preprocess.base import MultiplePreprocessorPipeline
from sub_clf.util.defaults import DEFAULTS
from sub_clf.util.io import FEATURE_LOADERS, load_texts
from sub_clf.util.utils import pretty_dumps


class Dataset:
    """Class for preprocessing and extracting features from comment data."""

    def __init__(self, config: Config) -> None:
        """
        Instantiate a `Dataset` class instance based on config file parameters.

        If `operation` == "preprocess":
            Read in raw data, apply preprocessing steps specified in config, and write
            output to .parquet.gz. In case of earlier partial failure, comments which
            were previously preprocessed are dropped to avoid redundant preprocessing.

        If `operation` == "extract":
            Read in preprocessed data, partition into train and test sets, extract
            features as specified in config, and write the resulting feature values to
            "train" and "test" subdirectories.

        If `operation` == "train":
            Raise `NotImplementedError`.

        Parameters
        ----------
        config : Config
            an object enumerating all parameters for your experiment

        Raises
        ------
        NotImplementedError
            if `operation` == "train"
        """

        output_writer = OutputWriter(config)
        output_writer.write_config()

        if config.operation == 'preprocess':
            # load preprocessing pipeline as specified in config
            pipeline = self.load_preprocessing_pipeline(config)
            pipeline.pipeline.steps.append((output_writer.name, output_writer))

            # load all raw data, group into batches by subreddit, and lazily preprocess
            raw_data = self.load_texts(config)

            # optionally "resume" by dropping samples that were already preprocessed
            if config.resume and (config.output_directory / 'data').exists():
                try:
                    raw_data = self.drop_preprocessed_comments(config, raw_data)
                except ValueError: # output directory has no preprocessed data files
                    pass

            batches = raw_data.groupby('subreddit')
            preprocessed_data = batches.apply(self.preprocess, pipeline=pipeline)

            # evaluate batched preprocessing (incl. writing preprocessed outputs)
            try:
                preprocessed_data.compute()
            finally:
                self.clean_up_tmp(config)

        elif config.operation == 'extract':
            # load feature extractor and preprocessed data as specified in config
            extractor = self.load_feature_extractor(config)
            preprocessed_data = self.load_texts(config)

            # partition preprocessed data into train and test sets
            partitions = self.partition_preprocessed_data(config, preprocessed_data)

            # fit feature extractor and vectorize train and test sets
            features = {
                'train' : extractor.fit_transform(partitions['train'].text),
                'test' : extractor.transform(partitions['test'].text)
            }

            # optionally rescale features as specified in config
            if config.scaler_pipeline is not None:
                scaler_pipeline = self.load_scaler_pipeline(config)
                # apply scaler pipeline to raw features

            # write features
            output_writer.write_comment_ids(partitions)
            output_writer.write_features(features)

        elif config.operation == 'train':
            raise NotImplementedError('`train` pipeline')


    def clean_up_tmp(self, config: Config) -> None:
        """Remove .tmp file containing preprocessed comment IDs."""

        ids_tmp_file = config.output_directory / 'data' / 'comment_ids.tmp'
        ids_tmp_file.unlink()


    def drop_preprocessed_comments(
        self,
        config: Config,
        raw_data: dd.core.DataFrame
    ) -> dd.core.DataFrame:
        """
        Resume partially complete preprocessing job by dropping samples were already
        preprocessed.
        """

        ids_tmp_file = config.output_directory / 'data' / 'comment_ids.tmp'
        preprocessed_comment_ids = ids_tmp_file.read_text().splitlines()

        raw_data = raw_data.loc[~raw_data.index.isin(preprocessed_comment_ids)]

        assert len(raw_data.index) > 0, (
            'All data in the following location(s) has already been preprocessed: '
            + str(config.raw_data_directory or config.raw_data_filepaths)
        )

        return raw_data


    def load_feature_extractor(self, config: Config):
        """Initialize `sklearn` vectorizer as specified in the config."""

        extractors = AVAILABLE['FEATURE_EXTRACTORS']
        extractor = extractors.get(config.extractor)

        if extractor is None:
            err = (f'"{config.extractor}" is not a recognized feature extractor. '
                   f'Please select one from the following: {pretty_dumps(extractors)}')
            raise KeyError(err)

        return extractor(**config.extractor_kwargs)


    def load_preprocessing_pipeline(self, config: Config) -> None:
        """Initialize preprocessor as specified in the config."""

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

        return pipeline


    def load_scaler_pipeline(self, config: Config) -> MultiplePreprocessorPipeline:
        """Load pipeline or other object responsible for postprocessing feature values."""

        raise NotImplementedError('Feature scaling')

        scaler_pipeline = self.load_scaler_pipeline(config)

        return {
            'train' : scaler_pipeline.fit_transform(train_features),
            'test' : scaler_pipeline.transform(test_features)
        }


    def load_texts(self, config: Config) -> dd.core.DataFrame:
        """Read and merge all .parquet.gz files in the specified directory/ies."""

        if config.operation == 'preprocess':
            path_or_paths = config.raw_data_directory or config.raw_data_filepaths

        elif config.operation == 'extract':
            path_or_paths = config.preprocessed_data_directory or config.preprocessed_data_filepaths

        return load_texts(path_or_paths)


    def partition_preprocessed_data(
        self,
        config: Config,
        preprocessed_data: dd.core.DataFrame
    ) -> Dict[str, dd.core.DataFrame]:
        """Randomly subsample all preprocessed comments into train and test sets."""

        int_ids = np.arange(len(preprocessed_data))
        train_ids, test_ids = train_test_split(int_ids, **config.train_test_split_kwargs)

        preprocessed_data['int_id'] = preprocessed_data.assign(i=1).i.cumsum() - 1

        return {
            'train' : preprocessed_data[preprocessed_data.int_id.isin(train_ids)],
            'test' : preprocessed_data[preprocessed_data.int_id.isin(test_ids)]
        }


    def preprocess(self, batch: dd.core.DataFrame, pipeline: Pipeline) -> dd.core.DataFrame:
        """Apply preprocessing pipeline to the raw data of a single subreddit."""

        if not batch.empty:
            return pipeline.preprocess(batch)
