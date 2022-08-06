"""
Collection of text preprocessers and a base class for assembling them into pipelines.
"""


import pandas as pd
import re
import spacy

from collections.abc import Sequence
from functools import partial
from nltk import stem
from operator import attrgetter
from overrides import overrides
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode
from typing import Optional

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

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        for strip_accents in [strip_accents_ascii, strip_accents_unicode]:
            data.text = data.text.map(strip_accents)
        return data


class CaseNormalizer(SinglePreprocessor):
    """
    Normalize comments to lowercase.

    E.g.:

    |Lorem ipsum DoloR sit amet.
        -->
    |lorem ipsum dolor sit amet.
    """

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        data.text = data.text.str.lower()
        return data


class PassthroughPreprocessor(SinglePreprocessor):
    """
    Pass raw text through unchanged, i.e., apply no preprocessing.

    E.g.:

    |Lorem ipsum dolor sit amet.
        -->
    |Lorem ipsum dolor sit amet.
    """

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        return data


class RegexTransformer(SinglePreprocessor):
    """
    Apply one or more `RegexTransformation` subclasses in parallel. The specific
    transformations are applied in the order passed when instantiating the class.
    """

    @overrides
    def __init__(self,
                 transformations: Sequence[RegexTransformation],
                 name: Optional[str] = None):
        super().__init__(name)

        self.transformations = transformations


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""

        patterns = []
        replacements = []

        for transformation_obj in self.transformations:
            for transformation in transformation_obj.transformations:
                pattern, replacement = transformation
                patterns.append(pattern)
                replacements.append(replacement)

        regex_transformations = {'to_replace' : patterns,
                                 'value' : replacements,
                                 'regex' : True}

        data.text = data.text.replace(**regex_transformations).str.strip()
        return data


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


    @overrides
    def __init__(self, type_='porter', **stem_method_kwargs):
        self.type_ = type_
        for key, value in stem_method_kwargs.items():
            setattr(self, key, value)
        super().__init__()

        stemmer = self.types.get(type_)
        if stemmer is None:
            raise TypeError(f'"{type_}" is not a recognized stemmer. Please select '
                            'one of the following: {pretty_dumps(self.types)}')
        self._stem = partial(stemmer().stem, **stem_method_kwargs)


    def stem(self, comment: str) -> str:
        return ' '.join(map(self._stem, comment.split()))


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        data.text = data.text.map(self.stem)
        return data


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


    @overrides
    def __init__(self, model='lg', lemmatize=False):
        self.model = model
        self.lemmatize = lemmatize
        super().__init__()

        model_ = self.models.get(model)
        if model_ is None:
            raise TypeError(f'"{model}" is not a recognized language model for '
                            'stop word removal. Please select one of the following: '
                            f'{pretty_dumps(self.models)}')
        self.nlp = spacy.load(model_)
        self.token = attrgetter('lemma_' if lemmatize else 'text')


    def remove_stopwords(self, comment: str) -> str:
        without_stopwords = (self.token(token) for token in self.nlp(comment)
                                               if not token.is_stop)
        return ' '.join(without_stopwords)


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        data.text = data.text.map(self.remove_stopwords)
        return data
