import pytest

from typing import Any, Dict
from unittest.mock import MagicMock

from sub_clf.experiment.config import Config, ConfigFileError


@pytest.mark.config
@pytest.mark.preprocess
@pytest.mark.extract
@pytest.mark.train
def test_confirm_parameter_value_dtypes():
    """Test the `Config._confirm_parameter_value_dtypes` method."""

    # mock `Config`
    config = MagicMock()
    config.operation = 'extract'
    config.extractor = None
    config.extractor_kwargs = {}
    config.output_directory = 'foo'
    config.overwrite_existing = False
    config.valid_fields_and_dtypes = {
        'extract' : {
            'extractor' : Any,
            'extractor_kwargs' : dict,
            'output_directory' : str,
            'overwrite_existing' : bool,
        }
    }

    # if method raises no exceptions, test passes
    Config._confirm_parameter_value_dtypes(config, 'extract')


@pytest.mark.config
@pytest.mark.preprocess
@pytest.mark.extract
@pytest.mark.train
@pytest.mark.parametrize(
    'operation,config_field_name,config_field_value,expected_error_message', [
        (
            # ID: "top-level dtype error"
            'preprocess',
            'preprocessors',
            False,
            '".+" is the incorrect data type'
        ),
        (
            # ID: "filepath not str"
            'preprocess',
            'raw_data_filepaths',
            [None],
            'All entries under ".+" must be'
        ),
        (
            # ID: "metrics entry too short"
            'train',
            'performance_metrics',
            [[]],
            'inner list has three items'
        ),
        (
            # ID: "metrics entry first arg not str"
            'train',
            'performance_metrics',
            [[(None, None, None)]],
            'a metric name'
        ),
        (
            # ID: "metrics entry second arg not str/null"
            'train',
            'performance_metrics',
            [[('', False, None)]],
            r'disambiguating suffix \(e.g., "macro" vs. "micro"'
        ),
        (
            # ID: "metrics entry third arg not dict"
            'train',
            'performance_metrics',
            [[('', None, None)]],
            'a dict of kwargs'
        ),
        (
            # ID: "preprocessor not dict"
            'preprocess',
            'preprocessors',
            [[]],
            r'Every entry under the ".+" field must be a single-item'
        ),
        (
            # ID: "preprocessor val not dict"
            'preprocess',
            'preprocessors',
            [{'' : None}],
            'dict of associated kwargs as the value'
        ),
        (
            # ID: "preprocessing pipeline too many"
            'preprocess',
            'preprocessing_pipeline',
            {0 : {}, 1 : {}},
            'Only a single preprocessing pipeline can be used'
        ),
        (
            # ID: "preprocessing pipeline val not dict"
            'preprocess',
            'preprocessing_pipeline',
            {0 : None},
            'The value of the ".+" field must be'
        )
    ],
    ids=[
        'top-level dtype error',
        'filepath not str',
        'metrics entry too short',
        'metrics entry first arg not str',
        'metrics entry second arg not str/null',
        'metrics entry third arg not dict',
        'preprocessor not dict',
        'preprocessor val not dict',
        'preprocessing pipeline too many',
        'preprocessing pipeline val not dict'
    ]
)
def test_confirm_parameter_value_dtypes_invalid(
    operation: str,
    config_field_name: str,
    config_field_value: Any,
    expected_error_message: str
):
    """Test the `Config._confirm_parameter_value_dtypes` method with invalid data types."""

    # mock `Config`
    config = MagicMock()
    config.operation = operation
    config.valid_fields_and_dtypes = Config.valid_fields_and_dtypes
    config._dict = {config_field_name : config_field_value}

    with pytest.raises(ConfigFileError, match=expected_error_message):
        Config._confirm_parameter_value_dtypes(config, operation)


@pytest.mark.config
@pytest.mark.preprocess
@pytest.mark.extract
@pytest.mark.train
def test_confirm_required_parameters_exist():
    """
    Test the `Config._confirm_required_parameters_exist` method when all required
    parameters are present.
    """

    # mock `Config`
    config = MagicMock()
    config._dict = {
        'output_directory' : None,
        'extractor' : None,
        'preprocessed_data_directory' : None
    }

    # if method raises no exceptions, test passes
    Config._confirm_required_parameters_exist(config, 'extract')


@pytest.mark.config
@pytest.mark.preprocess
@pytest.mark.extract
@pytest.mark.train
@pytest.mark.parametrize(
    'existing_parameters,expected_error_message', [
        (
            # ID: "missing single param"
            {'output_directory' : None},
            r'config file must contain a ".+" field\.'
        ),
        (
            # ID: "missing pair of params"
            {'output_directory' : None, 'extractor' : None},
            r'config file must contain a ".+" or ".+" field\.'
        ),
        (
            # ID: "conflicting pair of params"
            {
                'output_directory' : None,
                'extractor' : None,
                'preprocessed_data_directory' : None,
                'preprocessed_data_filepaths' : None
            },
            'config file must contain a ".+" or ".+" field, but not both'
        )
    ],
    ids=[
        'missing single param',
        'missing pair of params',
        'conflicting pair of params'
    ]
)
def test_confirm_required_parameters_exist_error(
    existing_parameters: Dict[str, None],
    expected_error_message: str
):
    """
    Test the `Config._confirm_required_parameters_exist` method when an issue with one
    or more required parameters is detected.
    """

    # mock `Config`
    config = MagicMock()
    config._dict = existing_parameters

    with pytest.raises(ConfigFileError, match=expected_error_message):
        Config._confirm_required_parameters_exist(config, 'extract')
