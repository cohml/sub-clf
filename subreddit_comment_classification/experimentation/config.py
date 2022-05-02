"""
Object that parses and validates a config file enumerating experimental parameters.
"""


import logging
import yaml

from pathlib import Path
from typing import Any, Dict


class Config:

    def __init__(self, config_filepath: Path) -> None:
        with config_filepath.open() as config_fh:
            self.dict = yaml.safe_load(config_fh)

        self._confirm_required_parameters_exist()
        self._confirm_parameter_value_dtypes()

        for parameter, value in self.dict.items():
            if parameter.endswith(('directory', 'filepath')):
                value = Path(value).resolve()
            elif parameter.endswith('filepaths'):
                value = [Path(v).resolve() for v in value]
            setattr(self, parameter, value)


    def __contains__(self, parameter) -> None:
        return parameter in self.dict


    def _confirm_parameter_value_dtypes(self) -> None:
        """Confirm all defined parameters are understood and of the correct types."""

        dtypes_by_field = {'raw_data_directory' : str,
                           'raw_data_filepaths' : list,
                           'features_directory' : str,
                           'features_filepaths' : list,
                           'preprocessors' : list,
                           'preprocessing_pipeline' : dict,
                           'write_preprocessed_data_filepath' : str,
                           'extractor' : Any,
                           'extractor_kwargs' : dict,
                           'write_features_filepath' : str,
                           'train_test_split_kwargs' : dict}

        err = None
        for field, dtype in dtypes_by_field.items():
            if field not in self.dict or field == 'extractor':
                continue

            if not isinstance(self.dict[field], dtype):
                err = f'"{field}" is the incorrect data type. Expected {dtype}.'

            elif field.endswith('paths'):
                for path in self.dict[field]:
                    if not isinstance(path, str):
                        err = f'All entries under "{field}" must be {str}.'

        if err is not None:
            raise TypeError(err)

        for field_name in ['preprocessors', 'preprocessing_pipeline']:
            field = self.dict.get(field_name)

            if field is None:
                continue

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
                           '"{<pipeline_name> : {<pipeline_verbosity>}}".')
                else:
                    preprocessing_pipeline_kwargs, = field.values()
                    if not isinstance(preprocessing_pipeline_kwargs, dict):
                        exception = TypeError
                        err = ('The value of the "preprocessing_pipeline" field must '
                               f'be {dict}.')

            if err is not None:
                raise exception(err)


    def _confirm_required_parameters_exist(self) -> None:
        """Confirm all requires parameters are defined in the config."""

        err = None
        conflicting_err = ('Your config file {} contain either a "{}" field or a "{}" '
                           'field, but not both.')
        missing_err = 'Your config file must contain a "{}" field.'

        if 'raw_data_directory' not in self.dict and 'raw_data_filepaths' not in self.dict:
            err = missing_err.format('raw_data_directory" or a "raw_data_filepaths')

        elif 'raw_data_directory' in self.dict and 'raw_data_filepaths' in self.dict:
            err = conflicting_err.format('must', 'raw_data_directory', 'raw_data_filepaths')

        elif 'preprocessors' not in self.dict and 'preprocessing_pipeline' not in self.dict:
            err = missing_err.format('preprocessors" or a "preprocessing_pipeline')

        elif 'preprocessors' in self.dict and 'preprocessing_pipeline' in self.dict:
            err = conflicting_err.format('must', 'preprocessors', 'preprocessing_pipeline')

        elif 'features_directory' in self.dict and 'features_filepaths' in self.dict:
            err = conflicting_err.format('may optionally', 'features_directory', 'features_filepath')

        else:
            required_fields = ['extractor', 'extractor_kwargs',
                               'train_test_split_kwargs']
            for required_field in required_fields:
                if required_field not in self.dict:
                    err = missing_err.format(required_field)
                    break

        if err is not None:
            raise ConfigFileError(err)


class ConfigFileError(Exception):
    pass
