"""
Abstract base classes for word embeddings vectorizers to subclass.
"""


import dask.dataframe as dd
import numpy as np

from types import ModuleType


class EmbeddingsVectorizer:

    def __init__(self, **kwargs):
        pass


    def build_embeddings_matrix(self) -> None:
        """Create matrix of pretrained word embeddings using a spacy model."""

        num_tokens = self.vocabulary.size.compute()
        embeddings_dim = self.nlp('The').vector.size
        embeddings_matrix_shape = (num_tokens, embeddings_dim)

        embeddings_matrix = np.zeros(embeddings_matrix_shape)
        for i, token in enumerate(self.vocabulary):
            embeddings_matrix[i] = self.nlp(token).vector

        return embeddings_matrix


    def fit_transform(self, preprocessed_data: dd.Series) -> np.ndarray:
        """
        Build up lexicon of unique tokens across all comments and use it to construct a
        matrix of pretrained word embeddings using a spacy model.
        """

        self.get_vocabulary(preprocessed_data)
        return self.build_embeddings_matrix()


    def get_vocabulary(self, preprocessed_data: dd.Series) -> None:                   ### METHOD MAY NEED REFINEMENT
        """
        Build lexicon of all unique tokens across all passed preprocessed comments.

        NOTE: THIS METHOD IS CURRENTLY VERY COARSE AND WILL NOT WORK UNLESS, AT A
              MINIMUM, ALL PUNCTUATION MARKS HAVE BEEN REMOVED.
        """

        self.vocabulary = (preprocessed_data.str.split()
                                            .explode()
                                            .unique()
                                            .sort_values())


    def load_model(self, model: ModuleType) -> None:
        """Load the spacy model to get embeddings from."""

        self.nlp = model.load()
