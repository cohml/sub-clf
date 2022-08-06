import pytest
import dask.dataframe as dd

from pathlib import Path

from sub_clf.util import io


@pytest.mark.preprocess
@pytest.mark.utils
@pytest.mark.parametrize(
    'raw_data_path_type', [
        'parent_directory',
        'subreddit_subdirectories'
    ],
)
def test_load_texts(
    inputs_directory_preprocessing: Path,
    raw_data_path_type: str,
    raw_data: dd.core.DataFrame):
    """Test the `util.io.load_texts` utility."""

    raw_data_paths = {
        'parent_directory' : inputs_directory_preprocessing / 'raw_texts',
        'subreddit_subdirectories' : [
            inputs_directory_preprocessing / 'raw_texts' / 'subreddit=github',
            inputs_directory_preprocessing / 'raw_texts' / 'subreddit=statistics'
        ]
    }

    computed_raw_data = io.load_texts(raw_data_paths[raw_data_path_type])

    expected_raw_data = raw_data

    dd.assert_eq(computed_raw_data, expected_raw_data)
