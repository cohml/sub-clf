import dask.dataframe as dd
import pandas as pd
import pytest

from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Union
from unittest.mock import MagicMock, Mock

from sklearn.feature_extraction.text import CountVectorizer

from sub_clf.experiment.dataset import Dataset


@pytest.mark.preprocess
def test_clean_up_tmp(
    inputs_directory_preprocessing: Path,
    preprocessing_outputs_directory_computed: Path
) -> None:
    """Test that `Dataset.clean_up_tmp` successfully removes .tmp comment IDs file."""

    # mock `Dataset`
    dataset = Mock()

    # mock `preprocess` config
    config = Mock()
    config.output_directory = preprocessing_outputs_directory_computed

    # mock .tmp file of preprocessed comment IDs
    source_filepath = inputs_directory_preprocessing / 'data' / 'comment_ids.tmp'
    target_filepath = config.output_directory / 'data' / 'comment_ids.tmp'
    target_filepath.parent.mkdir(exist_ok=True)
    copyfile(source_filepath, target_filepath)

    # just a safeguard that test setup worked
    assert target_filepath.exists()

    # test target method
    Dataset.clean_up_tmp(dataset, config)
    assert not target_filepath.exists()


@pytest.mark.preprocess
def test_drop_preprocessed_comments(
    inputs_directory_preprocessing: Path,
    raw_data,
    expected_raw_data_resumed: dd.core.DataFrame
):
    """Test `Dataset.drop_preprocessed_comments` method."""

    # mock `Dataset`
    dataset = Mock()

    # mock `preprocess` config
    config = Mock()
    config.output_directory = inputs_directory_preprocessing

    computed_raw_data_resumed = Dataset.drop_preprocessed_comments(dataset, config, raw_data)

    dd.assert_eq(
        computed_raw_data_resumed,
        expected_raw_data_resumed,
        check_categorical=False
    )


@pytest.mark.preprocess
def test_drop_preprocessed_comments_nothing_to_resume(
    inputs_directory_preprocessing: Path
):
    """
    Test `Dataset.drop_preprocessed_comments` method when all comments have already
    been preprocessed.
    """

    # mock `Dataset`
    dataset = Mock()

    # mock `preprocess` config
    config = Mock()
    config.output_directory = inputs_directory_preprocessing

    # mock raw data
    raw_data = dd.from_pandas(pd.DataFrame(), npartitions=1)

    expected_error_message = 'has already been preprocessed'

    with pytest.raises(AssertionError, match=expected_error_message):
        Dataset.drop_preprocessed_comments(dataset, config, raw_data)


@pytest.mark.extract
def test_load_feature_extractor():
    """Test `Dataset.load_feature_extractor` when a valid extractor is specified."""

    # mock `Dataset`
    dataset = Mock()

    # mock `extract` config
    config = Mock()
    config.extractor = 'CountVectorizer'
    config.extractor_kwargs = {}

    # get expected output
    expected_output = CountVectorizer()

    # compute actual output
    computed_output = Dataset.load_feature_extractor(dataset, config)

    # test same extractor type
    assert isinstance(computed_output, type(expected_output))

    # check same extractor parameters
    assert vars(computed_output) == vars(expected_output)


@pytest.mark.extract
def test_load_feature_extractor_invalid():
    """Test `Dataset.load_feature_extractor` when an invalid extractor is specified."""

    # mock `Dataset`
    dataset = Mock()

    # mock `extract` config
    config = Mock()
    config.extractor = 'NonexistentExtractor'
    config.extractor_kwargs = {}

    expected_error_message = 'not a recognized feature extractor'

    with pytest.raises(KeyError, match=expected_error_message):
        Dataset.load_feature_extractor(dataset, config)


@pytest.mark.preprocess
@pytest.mark.parametrize(
    'config_field_name,pipeline_or_preprocessor,contains', [
        ('preprocessing_pipeline', {'KitchenSinkPreprocessor' : {}}, True),
        ('preprocessors', [{'PassthroughPreprocessor' : {}}], False)
    ],
    ids=[
        'multi-preprocessor pipeline',
        'individual preprocessors'
    ]
)
def test_load_preprocessing_pipeline(
    config_field_name: str,
    pipeline_or_preprocessor: Union[Dict[str, Dict], List[Dict[str, Dict]]],
    contains: bool
):
    """Test `Dataset.load_preprocessing_pipeline` method."""

    # mock `Dataset`
    dataset = Mock()

    # mock `preprocess` config
    config = MagicMock()
    config.__contains__.return_value = contains
    setattr(config, config_field_name, pipeline_or_preprocessor)

    computed_result = Dataset.load_preprocessing_pipeline(dataset, config)

    assert hasattr(computed_result, 'preprocess')


@pytest.mark.preprocess
@pytest.mark.preprocess
@pytest.mark.parametrize(
    'config_field_name,pipeline_or_preprocessor,contains,expected_error_message', [
        (
            'preprocessing_pipeline',
            {'NonexistentPipeline' : {}},
            True,
            'not a recognized preprocessing pipeline'
        ),
        (
            'preprocessors',
            [{'NonexistentPreprocessor' : {}}],
            False,
            'not a recognized preprocessor'
        )
    ],
    ids=[
        'multi-preprocessor pipeline',
        'individual preprocessors'
    ]
)
def test_load_preprocessing_pipeline_invalid(
    config_field_name: str,
    pipeline_or_preprocessor: Union[Dict[str, Dict], List[Dict[str, Dict]]],
    contains: bool,
    expected_error_message: str
):
    """Test `Dataset.load_preprocessing_pipeline` when an invalid option is requested."""

    # mock `Dataset`
    dataset = Mock()

    # mock `preprocess` config
    config = MagicMock()
    config.__contains__.return_value = contains
    setattr(config, config_field_name, pipeline_or_preprocessor)

    with pytest.raises(KeyError, match=expected_error_message):
        Dataset.load_preprocessing_pipeline(dataset, config)


@pytest.mark.skip('Feature scaling not implemented')
@pytest.mark.extract
def test_load_scaler_pipeline():
    """Test `Dataset.load_scaler_pipeline` method."""
    pass


@pytest.mark.extract
def test_partition_preprocessed_data(
    preprocessed_data: dd.core.DataFrame,
    expected_partitions: Dict[str, dd.core.DataFrame]
):
    """Test `Dataset.partition_preprocessed_data` method."""

    # mock `Dataset`
    dataset = Mock()

    # mock `preprocess` config
    config = Mock()
    config.train_test_split_kwargs = {'random_state' : 0}

    computed_partitions = Dataset.partition_preprocessed_data(dataset, config, preprocessed_data)

    for partition_name, computed_partition in computed_partitions.items():
        expected_partition = expected_partitions[partition_name]

        dd.assert_eq(
            computed_partition,
            expected_partition,
            check_like=True,
            check_categorical=False
        )
