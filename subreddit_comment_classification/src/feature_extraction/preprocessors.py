"""
Collection of text preprocessers and a base class for assembling them into pipelines.
"""


import dask.dataframe as dd
import re

from string import punctuation, whitespace

from .abstract import SinglePreprocessor


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
