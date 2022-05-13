"""
Collection of text preprocessers and a base class for assembling them into pipelines.
"""


import dask.dataframe as dd
import re

from string import punctuation, whitespace
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode

from sub_clf.preprocess.base import SinglePreprocessor


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


class ApostropheNormalizer(SinglePreprocessor):
    """
    Normalize all apostrophes to a single standard form.

    E.g.:

    |Lorem ip‛sum dolor‘s sit amet
        -->
    |Lorem ip'sum dolor's sit amet
    """

    pattern = r"['‘’‛‚]"

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        apostrophes = {'pat' : re.compile(self.pattern),
                       'repl' : "'",
                       'regex' : True}
        return text.str.replace(**apostrophes)


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


class CodeBlockRemover(SinglePreprocessor):
    """
    Remove code blocks from comments, defined as lines beginning with four spaces or a
    literal tab character.

    E.g.:

    |Lorem ipsum dolor sit amet,
    |
    |    consectetur adipiscing elit,
    |    sed do eiusmod tempor incididunt
    |
    |ut labore et dolore magna aliqua.
        -->
    |Lorem ipsum dolor sit amet,
    |
    |
    |ut labore et dolore magna aliqua.
    """

    pattern = r'(^|\n)(\t| {4,})+.+?$'

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        code_block = {'pat' : re.compile(self.pattern, re.MULTILINE),
                      'repl' : '',
                      'regex' : True}
        return text.str.replace(**code_block)


class HyperlinkRemover(SinglePreprocessor):
    """
    Remove hyperlinks from comments.

    E.g.:

    |Lorem ipsum dolor sit amet, [consectetur](https://www.website.com) adipiscing elit
        -->
    |Lorem ipsum dolor sit amet, [consectetur]( adipiscing elit
    """

    pattern = r'http\S+'

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        hyperlink = {'pat' : re.compile(self.pattern),
                     'repl' : '',
                     'regex' : True}
        return text.str.replace(**hyperlink)


class InlineCodeRemover(SinglePreprocessor):
    """
    Remove inline code (i.e., "`code`") from comments.

    E.g.:

    |Lorem ipsum `dolor` sit amet
        -->
    |Lorem ipsum  sit amet
    """

    pattern = r'`.+?`'

    def transform(self, text: dd.Series) -> dd.Series:
        inline_code = {'pat' : re.compile(self.pattern),
                       'repl' : '',
                       'regex' : True}
        return text.str.replace(**inline_code)


class NewlineCollapser(SinglePreprocessor):
    """
    Collapse sequences of multiple newline characters into just one.

    E.g.:

    |Lorem ipsum dolor sit amet,
    |
    |
    |consectetur adipiscing elit
        -->
    |Lorem ipsum dolor sit amet,
    |consectetur adipiscing elit
    """

    pattern = r'\n{2,}'

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        consecutive_newlines = {'pat' : re.compile(self.pattern, re.MULTILINE),
                                'repl' : '\n',
                                'regex' : True}
        return text.str.replace(**consecutive_newlines)


class PunctuationRemover(SinglePreprocessor):
    """
    Remove common punctuation from comments. Note that apostrophes, hyphens, and
    common mathematical symbols are not removed.

    E.g.:

    |Here's "Lorem-ipsum. dolor sit @met, <<consectetur>> adipiscing elit!"
        -->
    |Here's Lorem-ipsum dolor sit met consectetur adipiscing elit
    """

    to_remove = ''
    to_keep = set("'-+*/=")
    for char in punctuation:
        if char not in to_keep:
            to_remove += char

    pattern = fr'[{to_remove}]'

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        punct = {'pat' : re.compile(self.pattern),
                 'repl' : '',
                 'regex' : True}
        return text.str.replace(**punct)


class QuoteRemover(SinglePreprocessor):
    """
    Remove quotation lines (i.e., starting with "> ") from comments.

    E.g.:

    |Lorem ipsum dolor sit amet,
    |
    |> consectetur adipiscing elit,
    |
    |>> sed do eiusmod tempor incididunt
    |
    |ut labore et dolore magna aliqua.
        -->
    |Lorem ipsum dolor sit amet,
    |
    |
    |
    |ut labore et dolore magna aliqua.
    """

    pattern = r'(^|\n)(&gt;|>).*?\n'    # NB: ">" is sometimes rendered as "&gt;"

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        quote = {'pat' : re.compile(self.pattern),
                 'repl' : '\n',
                 'regex' : True}
        return text.str.replace(**quote)


class Stemmer(SinglePreprocessor):
    """Stem comments."""

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        return text


class WhitespaceNormalizer(SinglePreprocessor):
    """
    Normalize all whitespace characters to the same form, then collapse sequences of
    consecutive whitespace down to a single whitespace character.

    E.g.:

    |Lorem   \t\t\n\n\t \t\n\r\x0b\x0c ipsum
        -->
    |Lorem ipsum
    """

    pattern = fr'[{whitespace}]+'

    def transform(self, text: dd.Series) -> dd.Series:
        """Apply preprocessing; required for any `SinglePreprocessor` subclass."""
        spaces = {'pat' : re.compile(self.pattern),
                  'repl' : ' ',
                  'regex' : True}
        return text.str.replace(**spaces)
