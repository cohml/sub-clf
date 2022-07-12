"""
Object that parses and validates a config file enumerating experimental parameters.
"""


import yaml

from pathlib import Path
from typing import Any, Dict

from sub_clf.util.defaults import DEFAULTS
from sub_clf.util.utils import full_path


class Config:

    def __init__(self, config_filepath: Path) -> None:
        with config_filepath.open() as config_fh:
            self.dict = yaml.safe_load(config_fh)

        operation = self._get_operation()
        self._confirm_required_parameters_exist(operation)
        self._confirm_parameter_value_dtypes(operation)

        self.dict = DEFAULTS['CONFIG'] | self.dict

        for parameter, value in self.dict.items():
            if parameter.endswith(('directory', 'file')) and value is not None:
                value = full_path(value)
            elif parameter.endswith('filepaths') and value is not None:
                value = [full_path(path) for path in value]
            setattr(self, parameter, value)


    def __contains__(self, parameter) -> None:
        return parameter in self.dict


    def _confirm_parameter_value_dtypes(self, operation: str) -> None:
        """
        Confirm all defined parameters are understood and of the correct types.

        Parameters
        ----------
        operation : str
            "preprocess", "extract", or "train"

        Raises
        ------
        ... # TODO
        """

        dtypes_by_field = {
            'extractor' : Any,
            'extractor_kwargs' : dict,
            'features_file' : str,
            'mode' : str,
            'model' : Any,
            'model_kwargs' : dict,
            'output_directory' : str,
            'overwrite_existing' : bool,
            'performance_metrics' : list,
            'preprocessors' : list,
            'preprocessing_pipeline' : dict,
            'raw_data_directory' : str,
            'raw_data_filepaths' : list,
            'save_features' : bool,         # <-- no longer optional; isolating feature extraction means you HAVE to save the features
            'save_model' : bool,
            'save_preprocessed_texts' : bool,    # <-- no longer optional; isolating preprocessing means you HAVE to save the preprocessed data
            'save_test_predictions' : bool,
            'save_train_test_ids' : bool,         # <-- no longer optional; isolating feature extraction means you HAVE to save the ids (b/c for certain features involving statistical transformations, you have to partition your data before computing features, so you have to track which samples are train and which are test; this is kind of a bummer since it means i'll have to extract new features whenever i want to try a differnt train-test split, but i don't see a way around it)
            'train_test_split_kwargs' : dict
        }




        if operation == 'preprocess':
            # required/optional keys:
                # output_directory
                # overwrite_existing
                # preprocessors
                # preprocessing_pipeline
                # raw_data_filepaths
                # raw_data_directory

        elif operation == 'extract':    # FYI - this is where the train-test splitting must occur
            # required/optional keys:
                # extractor
                # extractor_kwargs
                # output_directory
                # overwrite_existing
                # save_train_test_ids
                # train_test_split_kwargs

        elif operation == 'train':
            # required/optional keys:
                # features_file
                # mode (not yet implemented IIRC)
                # model
                # model_kwargs
                # output_directory
                # overwrite_existing
                # performance_metrics
                # save_model
                # save_test_predictions





        #### ORIGINAL TRAIN LOIGIC BELOW ####
        err = None
        for field, dtype in dtypes_by_field.items():
            if field not in self.dict or field in {'extractor', 'model'}:
                continue

            if not isinstance(self.dict[field], dtype):
                err = f'"{field}" is the incorrect data type. Expected {dtype}.'

            elif field.endswith('filepaths'):
                for filepath in self.dict[field]:
                    if not isinstance(filepath, str):
                        err = f'All entries under "{field}" must be {str}.'

        if err is not None:
            raise TypeError(err)

        for field_name in ['performance_metrics', 'preprocessors',
                           'preprocessing_pipeline']:
            field = self.dict.get(field_name)

            if field is None:
                continue

            elif field_name == 'performance_metrics':
                exception = TypeError
                performance_metrics_err = ('Your config\'s "performance_metrics" field '
                                           'must be a list of lists, where each inner '
                                           'list has three items: a metric name, a '
                                           'disambiguating suffix (e.g., "macro" vs. '
                                           '"micro", useful if the same metric is '
                                           'listed multiple times with different '
                                           'kwargs; set to `null` if no suffix), and '
                                           'a dict of kwargs (may be empty).')
                if not isinstance(self.dict[field_name], list):
                    err = performance_metrics_err
                    break
                for field in self.dict[field_name]:
                    if not len(field) == 3 or \
                       not isinstance(field[0], str) or \
                       not isinstance(field[1], (str, type(None))) or \
                       not isinstance(field[2], dict):
                        err = performance_metrics_err
                        break

            elif field_name == 'preprocessors':
                exception = TypeError
                preprocessor_err = ('All entries under your config\'s "preprocessors" '
                                    'field must be single-item {dict} with a valid '
                                    'preprocessor name string as the key and a dict '
                                    'of associated kwargs as the value.')

                for processor in field:
                    if not isinstance(processor, dict):
                        err = preprocessor_err
                        break
                    for processor_kwargs in processor.values():
                        if not isinstance(processor_kwargs, dict):
                            err = preprocessor_err
                            break

            elif field_name == 'preprocessing_pipeline':
                if len(field) != 1:
                    exception = ConfigFileError
                    err = ('Only a single preprocessing pipeline can be used. In your '
                           'config, please specify it as '
                           '"{<pipeline_name> : {verbose: <bool>}}".')
                else:
                    preprocessing_pipeline_kwargs, = field.values()
                    if not isinstance(preprocessing_pipeline_kwargs, dict):
                        exception = TypeError
                        err = ('The value of the "preprocessing_pipeline" field must '
                               f'be {dict}.')

            if err is not None:
                raise exception(err)


    def _confirm_required_parameters_exist(self, operation: str) -> None:
        """
        Confirm all requires parameters are defined in the config.

        Parameters
        ----------
        operation : str
            "preprocess", "extract", or "train"

        Raises
        ------
        ... # TODO
        """

        err = None
        conflicting_err = ('Your config file must contain either a "{}" field or a '
                           '"{}" field, but not both.')
        missing_err = 'Your config file must contain a "{}" field.'

        if 'raw_data_directory' not in self.dict and 'raw_data_filepaths' not in self.dict:
            err = missing_err.format('raw_data_directory" or a "raw_data_filepaths')

        elif 'raw_data_directory' in self.dict and 'raw_data_filepaths' in self.dict:
            err = conflicting_err.format('raw_data_directory', 'raw_data_filepaths')

        elif 'preprocessors' not in self.dict and 'preprocessing_pipeline' not in self.dict:
            err = missing_err.format('preprocessors" or a "preprocessing_pipeline')

        elif 'preprocessors' in self.dict and 'preprocessing_pipeline' in self.dict:
            err = conflicting_err.format('preprocessors', 'preprocessing_pipeline')

        else:
            required_fields = [
                'extractor',
                'performance_metrics',
                'output_directory',
                'mode',
                'model'
            ]
            for required_field in required_fields:
                if required_field not in self.dict:
                    err = missing_err.format(required_field)
                    break

        mode = self.dict['mode']
        if mode not in {'evaluate', 'train'}:
            err = ('The "mode" parameter must be set to either "train" or "evaluate". '
                   f'Got "{type(mode)}."')

        if err is not None:
            raise ConfigFileError(err)


    def _get_operation(self) -> str:
        """
        Identify and validate operation type specified in config.

        Returns
        -------
        operation : str
            "preprocess", "extract", or "train"

        Raises
        ------
        KeyError
            if config file has no "operation" field
        ValueError
            if the "operation" field's value is anything other than "preprocess",
            "extract", or "train"
        """

        err = None
        operation = self.dict.get('operation')

        if operation is None:
            exception = KeyError
            err = 'Your config must contain an "operation" field.'

        elif operation not in {'preprocess', 'extract', 'train'}:
            exception = ValueError
            err = ('Your config\'s "operation" field must specify either "preprocess",'
                   ' (to apply preprocessing to raw data), "extract" (to extract '
                   'features from preprocessed data), or "train" (to train a model).')

        if err is not None:
            raise exception(err)

        return operation


class ConfigFileError(Exception):
    pass
