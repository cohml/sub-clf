"""
Abstract base classes for text processing transformers to subclass.
"""


import dask
import dask.dataframe as dd
import warnings
warnings.simplefilter("ignore", UserWarning)

from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Pattern, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class RegexTransformation:
    """
    An abstract container for a single regex-based text transformation, designed for
    use with the `RegexTransformer` preprocessor.

    When instantiating the class, transformations must be passed in as a list of
    tuples. Each tuple must contain a precompiled regex pattern and the associated
    replacement string.
    """

    def __init__(self, _transformations: List[Tuple[Pattern, str]]):
        self.transformations = _transformations


    @property
    def transformations(self) -> List[Tuple[Pattern, str]]:
        return self._transformations


    @transformations.setter
    def transformations(self, _transformations: List[Tuple[Pattern, str]]):
        """
        Validate the data types of each transformations, which must be a 2-tuple. For
        each 2-tuple, the first item must be a precompiled regex and the second item
        must be a `str`.
        """

        err = 'The `{}` attribute of a `RegexTransformation` subclass must be a {}.'

        for transformation in _transformations:
            pattern, replacement = transformation

            if not isinstance(pattern, Pattern):
                raise TypeError(err.format('pattern', 'precompiled regex pattern'))
            elif not isinstance(replacement, str):
                raise TypeError(err.format('replacement', 'str'))

        self._transformations = _transformations


class SinglePreprocessor(BaseEstimator, TransformerMixin):
    """A single text preprocessing step."""

    def __init__(self, name: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__


    def __repr__(self):
        return f'{self.__module__}.{self.__class__.__name__}'


    def fit(self, X: dd.core.DataFrame, y: Optional[Any] = None):
        """This method must be defined for all `SinglePreprocessor` subclasses."""

        return self


    def preprocess(self, data: dd.core.DataFrame) -> dd.core.DataFrame:
        """
        Apply preprocessing step to comments.

        Parameters
        ----------
        data : dd.core.DataFrame
            data with comment texts

        Returns
        -------
        data : dd.core.DataFrame
            data with a single preprocessing step applied to comment texts
        """

        return self.fit_transform(data)


class MultiplePreprocessorPipeline:
    """
    A wrapper around skearn's `Pipeline` consisting of several text preprocessing steps.
    """

    def __init__(self,
                 *preprocessors: SinglePreprocessor,
                 verbose: bool = True):
        """
        Initialize a `MultiplePreprocessorPipeline` class instance with an arbitrary
        number of text preprocessing steps ready for use.

        Note: The preprocessing steps will be applied in the order in which they are
        passed into the constructor.

        Parameters
        ----------
        preprocessors : SinglePreprocessor
            sequence of text preprocessors to apply to the comments
        verbose : bool
            print status to stdout if True, else preprocess silently
        """

        steps = [(p.name, p) for p in preprocessors]
        self.pipeline = Pipeline(steps=steps,
                                 verbose=verbose,
                                 memory='cache_directory')


    def preprocess(self, data: dd.core.DataFrame, ncores: int = 1):
        """
        Apply parallelized pipeline of preprocessing steps to text.

        Parameters
        ----------
        data : dd.core.DataFrame
            data with comment texts
        ncores : int
            cores available for parallel computation

        Returns
        -------
        data : dd.core.DataFrame
            data with a single preprocessing step applied to comment texts
        """

        with dask.config.set(pool=ThreadPoolExecutor(max_workers=ncores)):
            return self.pipeline.fit_transform(data)
