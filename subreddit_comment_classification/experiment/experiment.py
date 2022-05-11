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
import torch
import yaml

from .available import AVAILABLE
from .config import Config
from .dataset import Dataset
from util.defaults import DEFAULTS
from util.io import FEATURE_SAVERS
from util.misc import pretty_dumps


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


    def evaluate_model(self):
        pass


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


    def try_pytorch(self):
        """Load Pytorch model."""
        # figure out pytorch models are loaded; currently i have zero clue

        raise NotImplemented('Pytorch models are not yet supported by this library.')


    def run(self): # main experiment logic here
        self.train_model()
        self.evaluate_model()
        self.write_report()


    def train_model(self):
        pass


    def generate_outputs(self) -> None:
        """
        Write all output files.

        Will include:
            - config (.json, .yaml)
            - report (.html, .ipynb)

        May include, depending on config:
            - model (binary)
            - train/test set comment IDs (.parquet.gz)
            - preprocessed comments texts (.parquet.gz)
            - feature values (.npz)
            - metadata (.csv)

        Raises
        ------
        FileExistsError
            if directory exists and `overwrite_existing` is `False` in (or absent
            from) your config
        """

        force = {'exist_ok' : self.config.overwrite_existing,
                 'parents' : True}

        # create top-level output directory
        output_directory = self.config.output_directory
        output_directory.mkdir(**force)

        # write config
        config_directory = output_directory / 'config'
        config_directory.mkdir(**force)
        config_output_filepath = config_directory / 'config'
        for config_saver, suffix in [(json.dump, '.json'), (yaml.safe_dump, '.yaml')]:
            with config_output_filepath.with_suffix(suffix).open('w') as fh:
                config_saver(self.config.dict, fh, indent=4, sort_keys=True)

        # write report
        pass # TODO

        # optionally write model
        if self.config.save_model:
            model_directory = output_directory / 'model'
            model_directory.mkdir(**force)
            pass # TODO

        if any([self.config.save_train_test_ids,
                self.config.save_preprocessed_texts,
                self.config.save_features,
                self.config.save_metadata]):
            data_directory = output_directory / 'data'
            data_directory.mkdir(**force)
            partitions = ['all', 'train', 'test']

        to_parquet_kwargs = DEFAULTS['IO']['TO_PARQUET_KWARGS'].copy()
        schema = to_parquet_kwargs['schema']
        for useless_kwarg in ['partition_on', 'schema']:
            to_parquet_kwargs.pop(useless_kwarg)

        # optionally write train/test comment IDs
        if self.config.save_train_test_ids:
            for partition in partitions:
                if partition == 'all':
                    continue
                partition_ids_directory = data_directory / partition / 'ids'
                partition_ids_directory.mkdir(**force)
                partition_ids = getattr(self.dataset, partition).ids
                (partition_ids.to_frame(name='comment_ids')
                              .to_parquet(partition_ids_directory,
                                          schema={'comment_id' : schema['comment_id']},
                                          **to_parquet_kwargs))

        # optionally write preprocessed comment texts
        if self.config.save_preprocessed_texts:
            for partition in partitions:
                preprocessed_text_directory = data_directory / partition / 'preprocessed_text'
                preprocessed_text_directory.mkdir(**force)
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

        # optionally write features
        if self.config.save_features:
            feature_saver = FEATURE_SAVERS[self.config.extractor]
            for partition in partitions:
                features_dir = data_directory / partition / 'features'
                features_dir.mkdir(**force)
                features_filepath = features_dir / 'features.npz'
                if partition == 'all':
                    features = self.dataset.features
                else:
                    features = getattr(self.dataset, partition).features
                feature_saver(features_filepath, features)

        return

        # optionally write metadata
        if self.config.save_metadata:
            for partition in partitions:
                metadata_directory = data_directory / partition / 'metadata'
                metadata_directory.mkdir(**force)
                metadata_filepath = metadata_directory / 'part_*.csv'
                if partition == 'all':
                    metadata = ... # TODO
                else:
                    metadata = ... # TODO
                metadata.to_csv(metadata_filepath,
                                index=False)
