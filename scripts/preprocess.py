"""
Collection of text preprocessers and a base class for assembling them into pipelines.
"""


import dask.dataframe as dd
import re
import warnings
warnings.simplefilter("ignore", UserWarning)

from overrides import overrides
from string import punctuation, whitespace
from typing import Any, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class SinglePreprocessor(BaseEstimator, TransformerMixin):
    """A single text preprocessing step."""

    @overrides
    def __init__(self):
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


class CaseNormalizer(SinglePreprocessor):
    """
    Normalize comments to lowercase.

    E.g.:

    |My name is Michelle. I come from Paris, France.
        -->
    |my name is michelle. i come from paris, france.
    """

    def transform(self, body: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        return body.str.lower()


class CodeBlockRemover(SinglePreprocessor):
    """
    Remove code blocks from comments, defined as lines beginning with four spaces or a
    literal tab character.

    E.g.:

    |Regular text 1.
    |
    |    code block
    |
    |Regular text 2.
        -->
    |Regular text 1.
    |
    |
    |Regular text 2.
    """

    pattern = r'(^|\n)(\t| {4,})+.+?$'

    def transform(self, body: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        code_block = {'pat' : re.compile(self.pattern, re.MULTILINE),
                      'repl' : '',
                      'regex' : True}
        return body.str.replace(**code_block)


class HyperlinkRemover(SinglePreprocessor):
    """
    Remove hyperlinks from comments.

    E.g.:

    |See [Wikipedia](https://www.wikipedia.org/) for more info.
        -->
    |See [Wikipedia]( for more info.
    """

    pattern = r'http\S+'

    def transform(self, body: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        hyperlink = {'pat' : re.compile(self.pattern),
                     'repl' : '',
                     'regex' : True}
        return body.str.replace(**hyperlink)


class InlineCodeRemover(SinglePreprocessor):
    """
    Remove inline code (i.e., "`code`") from comments.

    E.g.:

    |You need to import `numpy`.
        -->
    |You need to import .
    """

    pattern = r'`.+?`'

    def transform(self, body: dd.Series) -> dd.Series:
        inline_code = {'pat' : re.compile(self.pattern),
                       'repl' : '',
                       'regex' : True}
        return body.str.replace(**inline_code)


class NewlineCollapser(SinglePreprocessor):
    """
    Collapse sequences of multiple newline characters into just one.

    E.g.:

    |Here's the first line.
    |
    |
    |And here's the second.
        -->
    |Here's the first line.
    |And here's the second.
    """

    pattern = r'\n{2,}'

    def transform(self, body: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        consecutive_newlines = {'pat' : re.compile(self.pattern, re.MULTILINE),
                                'repl' : '\n',
                                'regex' : True}
        return body.str.replace(**consecutive_newlines)


class PunctuationRemover(SinglePreprocessor):
    """
    Remove common punctuation from comments. Note that apostrophes, hyphens, and
    common mathematical symbols are not removed.

    E.g.:

    |Here's an example. Of some, <<tip-top>> punctu@tion marks!
        -->
    |Here's an example Of some tip-top punctution marks
    """

    to_remove = ''
    to_keep = set("'-+*/=")
    for char in punctuation:
        if char not in to_keep:
            to_remove += char

    pattern = fr'[{to_remove}]'

    def transform(self, body: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        punct = {'pat' : re.compile(self.pattern),
                 'repl' : '',
                 'regex' : True}
        return body.str.replace(**punct)


class QuoteRemover(SinglePreprocessor):
    """
    Remove quotation lines (i.e., starting with "> ") from comments.

    E.g.:

    |Your question was:
    |
    |> Why?
    |
    |Because I said so.
        -->
    |Your question was:
    |
    |
    |Because I said so.
    """

    pattern = r'(^|\n)(&gt;|>).*?\n'    # NB: ">" is sometimes rendered as "&gt;"

    def transform(self, body: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        quote = {'pat' : re.compile(self.pattern),
                 'repl' : '\n',
                 'regex' : True}
        return body.str.replace(**quote)


class Stemmer(SinglePreprocessor):
    """Stem comments."""

    def transform(self, body: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        return body


class WhitespaceNormalizer(SinglePreprocessor):
    """
    Normalize all whitespace characters to the same form, then collapse sequences of
    consecutive whitespace down to a single whitespace character.

    E.g.:

    |Example   \t\t\n\n\t \t\n\r\x0b\x0c here.
        -->
    |Example here.
    """

    pattern = fr'[{whitespace}]+'

    def transform(self, body: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        spaces = {'pat' : re.compile(self.pattern),
                  'repl' : ' ',
                  'regex' : True}
        return body.str.replace(**spaces)


class MultiplePreprocessorPipeline:
    """
    A wrapper around skearn's `Pipeline` consisting of several text preprocessing steps.
    """

    def __init__(self,
                 *preprocessors: SinglePreprocessor,
                 **pipeline_kwargs: Any):
        """
        Initialize a `MultiplePreprocessorPipeline` class instance with an arbitrary
        number of text preprocessing steps ready for use.

        Note: The preprocessing steps will be applied in the order in which they are
        passed into the constructor.

        Parameters
        ----------
        preprocessors : SinglePreprocessor
            sequence of text preprocessors to apply to the comments
        pipeline_kwargs : Any
            any keyword argument that `sklearn.pipeline.Pipeline` will accept
        """

        preprocessors = [(p.__class__.__name__, p) for p in preprocessors]
        self.pipeline = Pipeline(steps=preprocessors,
                                 memory='cache_directory',
                                 **pipeline_kwargs)


    def preprocess(self, body: dd.Series):
        """
        Apply preprocessing steps to comments.

        Parameters
        ----------
        body : dd.Series
            raw comment texts

        Returns
        -------
        processed_body : dd.Series
            processed comment texts
        """

        return self.pipeline.fit_transform(body)


class KitchenSinkPreprocessor(MultiplePreprocessorPipeline):
    """Apply all available preprocesing steps to comments."""

    preprocessors = [CaseNormalizer(),
                     HyperlinkRemover(),
                     InlineCodeRemover(),
                     CodeBlockRemover(),
                     QuoteRemover(),
                     PunctuationRemover(),
                     WhitespaceNormalizer(),
                     NewlineCollapser(),
                     Stemmer()]


    @overrides
    def __init__(self, **pipeline_kwargs):
        super().__init__(*self.preprocessors, **pipeline_kwargs)
