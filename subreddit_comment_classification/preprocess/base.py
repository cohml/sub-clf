"""
Abstract base classes for text processing transformers to subclass.
"""


import dask
import dask.dataframe as dd
import warnings
warnings.simplefilter("ignore", UserWarning)

from concurrent.futures import ThreadPoolExecutor
from overrides import overrides
from typing import Any, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class SinglePreprocessor(BaseEstimator, TransformerMixin):
    """A single text preprocessing step."""

    @overrides
    def __init__(self, **kwargs):
        pass


    @overrides
    def __repr__(self):
        return f'{self.__module__}.{self.__class__.__name__}'


    def fit(self, X: dd.Series, y: Optional[Any] = None):
        """This method must be defined for all `SinglePreprocessor` subclasses."""
        return self


    def preprocess(self, body: dd.Series):
        """
        Apply preprocessing step to comments.

        Parameters
        ----------
        body : dd.Series
            raw comment texts

        Returns
        -------
        processed_body : dd.Series
            processed comment texts
        """

        return self.fit_transform(body)


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

        steps = [(p.__class__.__name__, p) for p in preprocessors]
        self.pipeline = Pipeline(steps=steps,
                                 verbose=verbose,
                                 memory='cache_directory')


    def preprocess(self, body: dd.Series, ncores: int = 1):
        """
        Apply parallelized pipeline of preprocessing steps to text.

        Parameters
        ----------
        body : dd.Series
            raw comment texts
        ncores : int
            cores available for parallel computation

        Returns
        -------
        processed_body : dd.Series
            processed comment texts
        """

        with dask.config.set(pool=ThreadPoolExecutor(max_workers=ncores)):
            return self.pipeline.fit_transform(body)
