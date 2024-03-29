"""
Object representing an entire end-to-end machine learning experiment. Accordingly,
this object will handle data loading, data preprocessing (optional), feature extraction
(optional), model testing and evaluation, and reporting out of the results.

For inspiration on the end-to-end process of text-based ML using sklearn, see this page:
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

For a great, accessible introduction to feature extraction from text data (including
word embeddings using spacy models), see this page:
https://towardsdatascience.com/pretrained-word-embeddings-using-spacy-and-keras-textvectorization-ef75ecd56360
"""


import dask.dataframe as dd
import json
import pandas as pd
import torch
import yaml

from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)

from sub_clf.experiment.available import AVAILABLE
from sub_clf.experiment.config import Config
from sub_clf.experiment.dataset import Dataset
from sub_clf.experiment.report import Report

from sub_clf.util.defaults import DEFAULTS
from sub_clf.util.io import FEATURE_SAVERS
from sub_clf.util.utils import pretty_dumps


class Experiment:

    def __init__(self: 'Experiment',
                 config: Config,
                 dataset: Dataset) -> None:
        """
        Initialize an end-to-end machine learning experiment.

        Parameters
        ----------
        config : Config
            an object enumerating all parameters for your experiment
        dataset : Dataset
            the data your model will be trained and evaluated on
        """

        self.config = config
        self.dataset = dataset
        self.model = self.load_model()


    def evaluate_model(self, **kwargs):
        """Generate predictions on the test set and quantify performance."""

        self.dataset.test.predictions = self.model.predict(self.dataset.test.features)
        self.dataset.test.probabilities = self.model.predict_proba(self.dataset.test.features)

        self.results = Results(self)

        for metric_name, suffix, metric_kwargs in self.config.performance_metrics:
            metric = AVAILABLE['METRICS'][metric_name]
            metric_name += f'_{suffix}' if suffix else ''
            metric_value = metric(y_true=self.dataset.test.labels,
                                  y_pred=self.dataset.test.predictions,
                                  **metric_kwargs)
            setattr(self.results, metric_name, metric_value)


    def load_model(self):
        """Load model. If not found among `sklearn` models, try Pytorch models."""

        model = AVAILABLE['MODELS']['SKLEARN'].get(self.config.model)

        if model is None:
            model = self.try_pytorch()

        if model is None:
            models = {**AVAILABLE['MODELS']['SKLEARN'], **AVAILABLE['MODELS']['SPACY']}
            err = (f'The requested model type "{self.config.model}" cannot be found. '
                   f'Plese choose from among the following models: {pretty_dumps(models)}')
            raise ValueError(err)

        return model(**self.config.model_kwargs)


    def run(self): # main experiment logic here

        if self.config.mode == 'train':
            self.train_model()

        elif self.config.mode == 'evaluate':  # TODO
            err = '"Evaluation" mode is not yet implemented.'
            raise NotImplementedError(err)

            # remember that i want to be able to bypass the preprocessing and/or extraction steps
            # if paths to already-existing files are provided via the config; implement that here
            raw_data = Dataset.load_raw_data(self.config)
            preprocessed_text = Dataset.preprocess_text(self.config, raw_data)
            features = Dataset.extract_features(self.config, preprocessed_text)

        self.evaluate_model()
        self.write_report()
        self.save_outputs()


    def train_model(self, **kwargs):
        """Fit model to a `Dataset`."""

        self.model.fit(X=self.dataset.train.features,
                       y=self.dataset.train.labels,
                       **kwargs)


    def try_pytorch(self):
        """Load Pytorch model."""
        # figure out pytorch models are loaded; currently i have zero clue

        err = 'Pytorch models are not yet supported by this library.'
        raise NotImplementedError(err)


    def save_outputs(self) -> None:
        """
        Write all output files.

        Will include:
            - config (.json, .yaml)
            - report (.html, .ipynb)

        May include, depending on config:
            - model (binary)
            - train/test set comment IDs (.parquet.gz)
            - preprocessed comments texts (.parquet.gz)
            - test set predictions (.parquet.gz)
            - feature values (.npz)
            - metadata (.csv)

        Raises
        ------
        FileExistsError
            if directory exists and `overwrite_existing` is `False` in (or absent
            from) your config
        """

        self.force = {'exist_ok' : self.config.overwrite_existing,
                      'parents' : True}

        # create top-level directory for all outputs
        output_directory = self.config.output_directory
        output_directory.mkdir(**self.force)

        self._save_config(output_directory)
        self._save_report(output_directory)

        if self.config.save_model:
            self._save_model(output_directory)

        # initialize directory for all data-oriented outputs, if needed
        if any([self.config.save_train_test_ids,
                self.config.save_preprocessed_texts,
                self.config.save_test_predictions,
                self.config.save_features,
                self.config.save_metadata]):
            data_directory = output_directory / 'data'
            data_directory.mkdir(**self.force)
            partitions = ['all', 'train', 'test']

            # set kwargs for writing .parquet.gz partitions, if needed
            if any([self.config.save_train_test_ids,
                    self.config.save_preprocessed_texts]):
                to_parquet_kwargs = DEFAULTS['IO']['TO_PARQUET_KWARGS'].copy()
                schema = to_parquet_kwargs['schema']
                for useless_kwarg in ['partition_on', 'schema']:
                    to_parquet_kwargs.pop(useless_kwarg)

        if self.config.save_train_test_ids:
            self._save_ids(data_directory, partitions, to_parquet_kwargs, schema)

        if self.config.save_preprocessed_texts:
            self._save_preprocessed_data(data_directory, partitions, to_parquet_kwargs)

        if self.config.save_test_predictions:
            self._save_test_set_predictions(data_directory, to_parquet_kwargs)

        if self.config.save_features:
            self._save_features(data_directory, partitions)

        if self.config.save_metadata:
            self._save_metadata(data_directory, partitions)


    def write_report(self):
        """Create `Report` object based on experimental results."""

        self.report = Report(self)


    def _save_config(self, output_directory: Path) -> None:
        """Save config parameters in .json and .yaml formats."""

        config_directory = output_directory / 'config'
        config_directory.mkdir(**self.force)
        config_output_filepath = config_directory / 'config'

        config_savers = {'.json' : json.dump,
                         '.yaml' : yaml.safe_dump}

        for suffix, config_saver in config_savers.items():
            with config_output_filepath.with_suffix(suffix).open('w') as fh:
                config_saver(self.config.dict, fh, indent=4, sort_keys=True)


    def _save_features(self, data_directory: Path, partitions: List[str]) -> None:
        """Save features to .npz file."""

        feature_saver = FEATURE_SAVERS[self.config.extractor]

        for partition in partitions:
            features_dir = data_directory / partition / 'features'
            features_dir.mkdir(**self.force)
            features_filepath = features_dir / 'features.npz'

            if partition == 'all':
                features = self.dataset.features
            else:
                features = getattr(self.dataset, partition).features

            feature_saver(features_filepath, features)


    def _save_ids(self,
                  data_directory: Path,
                  partitions: List[str],
                  to_parquet_kwargs: Dict[str, Any],
                  schema: Dict[str, Any]
                  ) -> None:
        """Save unique comment IDs for the train and test set partitions."""

        for partition in partitions:
            if partition == 'all':
                continue

            partition_ids_directory = data_directory / partition / 'ids'
            partition_ids_directory.mkdir(**self.force)
            partition_ids = getattr(self.dataset, partition).ids
            (dd.from_array(partition_ids)
               .to_frame(name='comment_id')
               .to_parquet(partition_ids_directory,
                           schema={'comment_id' : schema['comment_id']},
                           **to_parquet_kwargs))


    def _save_metadata(self, data_directory: Path, partitions: List[str]) -> None:
        """Save metadata to .csv."""

        for partition in partitions:
            metadata_directory = data_directory / partition / 'metadata'
            metadata_directory.mkdir(**self.force)
            metadata_filepath = metadata_directory / 'part_*.csv' # TODO

            if partition == 'all':
                metadata = ... # TODO
            else:
                metadata = ... # TODO

            metadata.to_csv(metadata_filepath, index=False)


    def _save_model(self, output_directory: Path) -> None:
        """Save model to binary file."""

        return # TODO

        model_directory = output_directory / 'model'
        model_directory.mkdir(**self.force)


    def _save_preprocessed_data(self,
                                data_directory: Path,
                                partitions: List[str],
                                to_parquet_kwargs: Dict[str, Any]
                                ) -> None:
        """Save preprocessed comment texts to .parquet.gz partitions."""

        for partition in partitions:
            preprocessed_text_directory = data_directory / partition / 'preprocessed_text'
            preprocessed_text_directory.mkdir(**self.force)

            if partition == 'all':
                preprocessed_text = self.dataset.preprocessed_text
            else:
                partition_ids = getattr(self.dataset, partition).ids
                preprocessed_text = dd.from_array(self.dataset
                                                      .preprocessed_text
                                                      .compute()
                                                      .loc[partition_ids])

            (preprocessed_text.to_frame(name='text')
                              .to_parquet(preprocessed_text_directory,
                                          **to_parquet_kwargs))


    def _save_report(self, output_directory: Path) -> None:
        """Write all results and analyses to .ipynb and an equivalent HTML file."""

        self.report.save(output_directory)


    def _save_test_set_predictions(self,
                                   data_directory: Path,
                                   to_parquet_kwargs: Dict[str, Any]
                                   ) -> None:
        """Write all test set predictions to .parquet.gz partitions."""

        test_predictions_directory = data_directory / 'test' / 'predictions'
        test_predictions_directory.mkdir(**self.force)

        y_true_categorical = dd.from_array(self.dataset.test.labels)
        y_true_categorical = y_true_categorical.map(self.dataset.class_name_mappings)

        y_pred_categorical = dd.from_array(self.dataset.test.predictions)
        y_pred_categorical = y_pred_categorical.map(self.dataset.class_name_mappings)

        is_correct = self.dataset.test.labels == self.dataset.test.predictions
        is_correct = dd.from_array(is_correct).astype(int)

        ids = dd.from_array(self.dataset.test.ids)

        (dd.concat([y_true_categorical, y_pred_categorical, is_correct], axis=1)
           .rename(columns={0 : 'y_true', 1 : 'y_pred', 2 : 'is_correct'})
           .set_index(ids)
           .to_parquet(test_predictions_directory, **to_parquet_kwargs))


class Results:

    def __init__(self, experiment: Experiment) -> None:
        labels = experiment.dataset.test.labels
        predictions = experiment.dataset.test.predictions
        class_names = experiment.dataset.class_name_mappings.values()

        prfs = precision_recall_fscore_support(labels, predictions)
        prfs = dict(zip(class_names, zip(*prfs)))
        columns = ['precision', 'recall', 'f1-score', 'support']
        self.classification_report_df = pd.DataFrame(prfs, index=columns).T

        self.classification_report = classification_report(labels, predictions,
                                                           target_names=class_names)


    def __repr__(self):
        return self.classification_report
