"""
Object that parses and validates a config file enumerating parameters for the requested
operation.
"""


import yaml

from pathlib import Path
from typing import Any, Dict

from sub_clf.util.defaults import DEFAULTS
from sub_clf.util.utils import full_path


class Config:

    # all valid config file fields, both required and optional, broken out by operation
    valid_fields_and_dtypes = {
        'preprocess' : {
            'output_directory' : str,
            'overwrite_existing' : bool,
            'preprocessors' : list,
            'preprocessing_pipeline' : dict,
            'raw_data_directory' : str,
            'raw_data_filepaths' : list
        },
        'extract' : {
            'extractor' : Any,
            'extractor_kwargs' : dict,
            'preprocessed_data_directory' : str,
            'output_directory' : str,
            'overwrite_existing' : bool,
            'train_test_split_kwargs' : dict
        },
        'train' : {
            'features_file' : str,
            'model' : Any,
            'model_kwargs' : dict,
            'output_directory' : str,
            'overwrite_existing' : bool,
            'performance_metrics' : list,
            'save_model' : bool,
            'save_test_predictions' : bool
        }
    }


    def __init__(self, config_filepath: Path, operation: str) -> None:
        with config_filepath.open() as config_fh:
            self.dict = yaml.safe_load(config_fh)

        self._confirm_required_parameters_exist(operation)
        self._confirm_parameter_value_dtypes(operation)

        self.dict = DEFAULTS['CONFIG'] | self.dict

        for parameter, value in self.dict.items():
            if parameter.endswith(('directory', 'file')) and value is not None:
                value = full_path(value)
            elif parameter.endswith('filepaths') and value is not None:
                value = [full_path(path) for path in value]
            setattr(self, parameter, value)

        self.operation = operation


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
        ConfigFileError
            if a field in the config file contains an unexpected datatype
        """

        for field, dtype in self.valid_fields_and_dtypes[operation].items():

            if field not in self.dict or dtype is Any:
                continue

            if not isinstance(self.dict[field], dtype):
                raise ConfigFileError(
                    f'"{field}" is the incorrect data type. Expected {dtype}.'
                )

            if field.endswith('filepaths'):
                for filepath in self.dict[field]:
                    if not isinstance(filepath, str):
                        raise ConfigFileError(
                            f'All entries under "{field}" must be {str}.'
                        )

            elif field == 'performance_metrics':
                for metric in self.dict[field]:
                    if not len(metric) == 3 or \
                       not isinstance(metric[0], str) or \
                       not isinstance(metric[1], (str, type(None))) or \
                       not isinstance(metric[2], dict):
                        raise ConfigFileError(
                            f'The "{field}" field  must be a list of lists where each '
                            'inner list has three items: a metric name, a '
                            'disambiguating suffix (e.g., "macro" vs. "micro", useful '
                            'if the same metric is listed multiple times with '
                            'different kwargs; set to `null` if no suffix), and a '
                            'dict of kwargs (may be empty).'
                        )

            elif field == 'preprocessors':
                err = (
                    f'Every entry under the "{field}" field must be a single-item '
                    f'{dict} with a valid preprocessor name string as the key and a '
                    'dict of associated kwargs as the value.'
                )
                for preprocessor in self.dict[field]:
                    if not isinstance(preprocessor, dict):
                        raise ConfigFileError(err)
                    for preprocessor_kwargs in preprocessor.values():
                        if not isinstance(preprocessor_kwargs, dict):
                            raise ConfigFileError(err)

            elif field == 'preprocessing_pipeline':
                if len(self.dict[field]) != 1:
                    raise ConfigFileError(
                        'Only a single preprocessing pipeline can be used. In your '
                        'config file, it must specified as "{<pipeline_name> : '
                        '{verbose: <bool>}}".'
                    )
                preprocessing_pipeline_kwargs, = self.dict[field].values()
                if not isinstance(preprocessing_pipeline_kwargs, dict):
                    raise ConfigFileError(
                        f'The value of the "{field}" field must be {dict}.'
                    )


    def _confirm_required_parameters_exist(self, operation: str) -> None:
        """
        Confirm all required parameters are defined in the config file.

        Parameters
        ----------
        operation : str
            "preprocess", "extract", or "train"

        Raises
        ------
        ConfigFileError
            - if the config file is missing any required fields
            - if the config file contains any mutually exclusive fields
        """

        def check_conflicting(field_1, field_2):
            """Check whether the config file contains two mutually exclusive fields."""

            if field_1 in self.dict and field_2 in self.dict:
                raise ConfigFileError(
                    f'Your config file must contain a "{field_1}" or "{field_2}" field, '
                    'but not both.'
                )


        def check_missing(field_1, field_2=None):
            """Check whether required fields are missing from the config file."""

            if field_2 is not None:
                if field_1 not in self.dict and field_2 not in self.dict:
                    raise ConfigFileError(
                        f'Your config file must contain a "{field_1}" or "{field_2}" field.'
                    )
            else:
                if field_1 not in self.dict:
                    raise ConfigFileError(
                        f'Your config file must contain a "{field_1}" field.'
                    )


        # this field is required regsrdless of requested operation
        check_missing('output_directory')

        if operation == 'preprocess':
            check_missing('preprocessors', 'preprocessing_pipeline')
            check_conflicting('preprocessors', 'preprocessing_pipeline')

            check_missing('raw_data_directory', 'raw_data_filepaths')
            check_conflicting('raw_data_directory', 'raw_data_filepaths')

        elif operation == 'extract':
            check_missing('extractor')
            check_missing('preprocessed_data_directory')

        elif operation == 'train':
            check_missing('features_file')
            check_missing('model')
            check_missing('performance_metrics')


class ConfigFileError(Exception):
    """Generic exception for all malformed config file errors."""
    pass
