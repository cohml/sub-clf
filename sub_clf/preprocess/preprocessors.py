"""
Collection of text preprocessers and a base class for assembling them into pipelines.
"""


import dask.dataframe as dd
import re
import spacy

from collections.abc import Sequence
from functools import partial
from nltk import stem
from operator import attrgetter
from string import punctuation, whitespace
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode

from sub_clf.preprocess.base import RegexTransformation, SinglePreprocessor
from sub_clf.util.utils import pretty_dumps


class AccentRemover(SinglePreprocessor):
    """
    Remove all ASCII and Unicode accent marks from comments.

    E.g.:

    |ìíîïñòóôõöùúûüý and \xec\xed\xee\xef\xf1\xf2\xf3\xf4\xf5\xf6\xf9\xfa\xfb\xfc\xfd
        ->
    |iiiinooooouuuuy and iiiinooooouuuuy
    """

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        for strip_accents in [strip_accents_ascii, strip_accents_unicode]:
            text = text.map(strip_accents)
        return text


class CaseNormalizer(SinglePreprocessor):
    """
    Normalize comments to lowercase.

    E.g.:

    |Lorem ipsum DoloR sit amet.
        -->
    |lorem ipsum dolor sit amet.
    """

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        return text.str.lower()


class PassthroughPreprocessor(SinglePreprocessor):
    """
    Pass raw text through unchanged, i.e., apply no preprocessing.

    E.g.:

    |Lorem ipsum dolor sit amet.
        -->
    |Lorem ipsum dolor sit amet.
    """

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        return text


class RegexTransformer(SinglePreprocessor):
    """
    Apply one or more `RegexTransformation` subclasses in parallel. The specific
    transformations are applied in the order passed when instantiating the class.
    """

    def __init__(self, transformations: Sequence[RegexTransformation]):
        self.transformations = transformations


    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""

        transformations = {}
        for transformation in self.transformations:
            transformations.update(transformation.transformation)

        return text.replace(transformations, regex=True)


class Stemmer(SinglePreprocessor):  # note: not used; tokens are lemmatized instead (by `StopwordRemover`)
    """
    Stem comments.

    The following NLTK stemers are currently supported:
    - LancasterStemmer ('lancaster')
        - https://www.nltk.org/api/nltk.stem.lancaster.html
    - PorterStemmer ('porter')
        - https://www.nltk.org/api/nltk.stem.porter.html
    - RegexpStemmer ('regexp')
        - https://www.nltk.org/api/nltk.stem.regexp.html
    - SnowballStemmer ('snowball')
        - https://www.nltk.org/api/nltk.stem.snowball.html
    """

    types = {'lancaster' : stem.lancaster.LancasterStemmer,
             'porter' : stem.porter.PorterStemmer,
             'regexp' : stem.regexp.RegexpStemmer,
             'snowball' : stem.snowball.SnowballStemmer}


    def __init__(self, type_='porter', **stem_method_kwargs):
        stemmer = self.types.get(type_)
        if stemmer is None:
            raise TypeError(f'"{type_}" is not a recognized stemmer. Please select '
                            'one of the following: {pretty_dumps(self.types)}')
        self._stem = partial(stemmer().stem, **stem_method_kwargs)


    def stem(self, comment: str):
        return ' '.join(map(self._stem, comment.split()))


    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        return text.map(self.stem, meta=('text', 'object'))


class StopwordRemover(SinglePreprocessor):
    """
    Remove stopwords from comments.

    E.g.:

    |This sentence contains two stop words.
        -->
    |sentence contains stop words .  # `lemmatize=False` (default)
    |sentence contain stop word .    # `lemmatize=True`

    The following spaCy language models are currently supported for stop word removal:
    - en_core_web_lg ('lg')
        - https://spacy.io/models/en#en_core_web_lg
    - en_core_web_trf ('trf')
        - https://spacy.io/models/en#en_core_web_trf
    """

    models = {'lg' : 'en_core_web_lg',
              'trf' : 'en_core_web_trf'}


    def __init__(self, model='lg', lemmatize=False):
        model_ = self.models.get(model)
        if model_ is None:
            raise TypeError(f'"{model}" is not a recognized language model for '
                            'stop word removal. Please select one of the following: '
                            f'{pretty_dumps(self.models)}')
        self.nlp = spacy.load(model_)
        self.token = attrgetter('lemma_' if lemmatize else 'text')


    def remove_stopwords(self, comment: str):
        without_stopwords = (self.token(token) for token in self.nlp(comment)
                                               if not token.is_stop)
        return ' '.join(without_stopwords)


    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        return text.map(self.remove_stopwords, meta=('text', 'object'))
