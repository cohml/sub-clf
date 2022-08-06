import dask.dataframe as dd
import pytest

from pathlib import Path
from shutil import rmtree
from unittest.mock import Base, Mock

from sub_clf.experiment.config import Config
from sub_clf.experiment.writer import OutputWriter
from sub_clf.util.defaults import DEFAULTS


@pytest.mark.writer
@pytest.mark.preprocess
@pytest.mark.extract
@pytest.mark.train
def test_write_config(
    outputs_directory_computed: Path,
    outputs_directory_expected: Path
):
    """Test the `OutputWriter.write_config` method."""

    # mock `Config`
    config = Base()
    config.operation = 'preprocess'
    config.output_directory = outputs_directory_computed
    config.overwrite_existing = True
    config.raw_data_filepaths = [Path('foo')]

    outputs = {
        'json' : {
            'computed' : outputs_directory_computed / 'configs' / 'config.json',
            'expected' : outputs_directory_expected / 'configs' / 'config.json'
        },
        'yaml' : {
            'computed' : outputs_directory_computed / 'configs' / 'config.yaml',
            'expected' : outputs_directory_expected / 'configs' / 'config.yaml'
        }
    }

    # just a safeguard that test is setup properly
    for filetype in outputs:
        computed_output = outputs[filetype]['computed']
        assert not computed_output.exists()

    try:
        OutputWriter(config).write_config()

        for filetype in outputs:
            computed_output = outputs[filetype]['computed']
            assert computed_output.exists()

            expected_output = outputs[filetype]['expected']

            # slightly normalize the text lest IDE silent autoformatting interfere
            computed_text = computed_output.read_text().replace('\t', '    ').strip()
            expected_text = expected_output.read_text().replace('\t', '    ').strip()

            assert computed_text == expected_text

    finally:
        for filetype in outputs:
            computed_output = outputs[filetype]['computed']
            computed_output.unlink(missing_ok=True)


@pytest.mark.writer
@pytest.mark.extract
def test_write_comment_ids(
    outputs_directory_computed: Path,
    outputs_directory_expected: Path,
    preprocessed_data: dd.core.DataFrame
):
    """Test the `OutputWriter.write_comment_ids` method."""

    # mock `Config`
    config = Mock()
    config.operation = 'preprocess'
    config.output_directory = outputs_directory_computed
    config.overwrite_existing = True

    outdirs = {
        'computed' : outputs_directory_computed / 'ids' / 'train',
        'expected' : outputs_directory_expected / 'ids' / 'train'
    }

    # just a safeguard that test is setup properly
    outdirs['computed'].mkdir(exist_ok=True, parents=True)
    computed_outputs = outdirs['computed'].iterdir()
    assert not list(computed_outputs)

    partitions = {'train' : preprocessed_data}

    try:
        OutputWriter(config).write_comment_ids(partitions)
        computed_output = dd.read_parquet(
            outdirs['computed'],
            **DEFAULTS['IO']['READ_PARQUET_KWARGS']
        )
        expected_output = preprocessed_data.index.to_frame(name='')

        dd.assert_eq(computed_output, expected_output)

    finally:
        for file in outdirs['computed'].iterdir():
            file.unlink()


@pytest.mark.writer
@pytest.mark.preprocess
def test_write_preprocessed_data(
    preprocessing_outputs_directory_computed: Path,
    preprocessed_data: dd.core.DataFrame
):
    """Test the `OutputWriter.write_preprocessed_data` method."""

    # mock `Config`
    config = Mock()
    config.operation = 'preprocess'
    config.output_directory = preprocessing_outputs_directory_computed
    config.overwrite_existing = True

    try:
        OutputWriter(config).write_preprocessed_data(preprocessed_data.compute())

        # check preprocessed comment IDs .tmp file
        computed_output_ids_file = preprocessing_outputs_directory_computed / 'data' / 'comment_ids.tmp'
        assert computed_output_ids_file.exists()
        computed_output_ids = set(computed_output_ids_file.read_text().splitlines())
        computed_output_ids_file.unlink(missing_ok=True)
        expected_output_ids = set(preprocessed_data.index)
        assert computed_output_ids == expected_output_ids

        # check preprocessed data
        computed_output_data = dd.read_parquet(
            preprocessing_outputs_directory_computed / 'data',
            **DEFAULTS['IO']['READ_PARQUET_KWARGS']
        )
        expected_output_data = preprocessed_data
        dd.assert_eq(computed_output_data, expected_output_data)

    finally:
        for subreddit_subdirectory in (preprocessing_outputs_directory_computed / 'data').iterdir():
            rmtree(subreddit_subdirectory)
