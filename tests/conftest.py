import dask.dataframe as dd
import pytest
from pathlib import Path
from typing import Dict

from sub_clf.util.defaults import DEFAULTS


@pytest.fixture(scope='module')     # possible to mix fixtures with classes? like, could i have a "TestDirectory" class of fixtures, with this and all other fixtures defined so far (all dirs) being members?
def fixtures_directory() -> Path:
    package_root_directory = Path(__file__).resolve().parent.parent
    return package_root_directory / 'tests' / 'fixtures'


@pytest.fixture(scope='module')
def inputs_directory(fixtures_directory) -> Path:
    return fixtures_directory / 'input'


@pytest.fixture(scope='module')
def inputs_directory_preprocessing(inputs_directory) -> Path:
    return inputs_directory / 'preprocess'


@pytest.fixture(scope='module')
def inputs_directory_extract(inputs_directory) -> Path:
    return inputs_directory / 'extract'


@pytest.fixture(scope='module')
def outputs_directory_computed(fixtures_directory) -> Path:
    return fixtures_directory / 'output' / 'computed'


@pytest.fixture(scope='module')
def outputs_directory_expected(fixtures_directory) -> Path:
    return fixtures_directory / 'output' / 'expected'


@pytest.fixture(scope='module')
def preprocessing_outputs_directory_computed(outputs_directory_computed) -> Path:
    directory = outputs_directory_computed / 'preprocess'
    directory.mkdir(exist_ok=True, parents=True)
    return directory


@pytest.fixture(scope='module')
def raw_data(inputs_directory_preprocessing) -> dd.core.DataFrame:
    return dd.read_parquet(
        inputs_directory_preprocessing / 'raw_texts',
        **DEFAULTS['IO']['READ_PARQUET_KWARGS']
    )


@pytest.fixture(scope='module')
def expected_raw_data_resumed(inputs_directory_preprocessing) -> dd.core.DataFrame:
    return dd.read_parquet(
        inputs_directory_preprocessing / 'raw_texts_resumed',
        **DEFAULTS['IO']['READ_PARQUET_KWARGS']
    )


@pytest.fixture(scope='module')
def preprocessed_data(inputs_directory_extract) -> dd.core.DataFrame:
    """Note: `preprocessed_data` == `raw_data` (fixture above) + `CaseNormalizer`."""
    return dd.read_parquet(
        inputs_directory_extract / 'preprocessed_texts',
        **DEFAULTS['IO']['READ_PARQUET_KWARGS']
    )


@pytest.fixture(scope='module')
def expected_preprocessed_data_kitchen_sink(
    inputs_directory_preprocessing
) -> dd.core.DataFrame:
    return dd.read_parquet(
        inputs_directory_preprocessing / 'preprocessed_texts_ksp',
        **DEFAULTS['IO']['READ_PARQUET_KWARGS']
    )


@pytest.fixture(scope='module')
def expected_partitions(inputs_directory_extract) -> Dict[str, dd.core.DataFrame]:
    partitions = {}
    for train_test in ['train', 'test']:
        partition = dd.read_parquet(
            inputs_directory_extract / 'partitions' / train_test,
            **DEFAULTS['IO']['READ_PARQUET_KWARGS']
        )
        partitions[train_test] = partition
    return partitions
