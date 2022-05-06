"""
Feature extractors for building up matrices of pretrained word embeddings using spacy
models "en_core_web_lg" and "en_core_web_trf".
"""


import dask.dataframe as dd
import numpy as np

import en_core_web_lg
import en_core_web_trf

from overrides import overrides

from embed.base import EmbeddingsVectorizer


class LgEmbeddingsVectorizer(EmbeddingsVectorizer):

    @overrides
    def fit_transform(self, v: dd.Series) -> np.ndarray:
        """
        Use spacy's `en_core_web_lg` language model to construct a matrix of pretrained
        word embeddings for the passed preprocessed comments.

        Parameters
        ----------
        preprocessed_text : dd.Series
            preprocessed comment texts

        Returns
        -------
        embedding_matrix : np.ndarray
            matrix of pre-trained word embeddings for the preprocessed comment texts
        """

        self.load_model(en_core_web_lg)
        embeddings_matrix = super().fit_transform(preprocessed_text)
        return embeddings_matrix


class TrfEmbeddingsVectorizer(EmbeddingsVectorizer):

    @overrides
    def fit_transform(self, preprocessed_text: dd.Series) -> np.ndarray:
        """
        Use spacy's `en_core_web_lg` language model to construct a matrix of pretrained
        word embeddings for the passed preprocessed comments.

        Parameters
        ----------
        preprocessed_text : dd.Series
            preprocessed comments raw text

        Returns
        -------
        embeddings_matrix : np.ndarray
            matrix of pre-trained word embeddings for the passed preprocessed comments
        """

        # for more info on this error, see:
        # https://github.com/explosion/spaCy/discussions/7643
        err = ('The `en_core_web_trf` model works a bit differently from the '
               '`en_core_web_lg` model. Crucially, it does not ship with pretrained '
               'word embedding vectors. I need to look more into why this is and how '
               'to get around it before `en_core_web_trf` will be available for use.')
        raise NotImplementedError(err)

        # self.load_model(en_core_web_trf)
        # embeddings_matrix = super().fit_transform(preprocessed_text)
        # return embeddings_matrix
