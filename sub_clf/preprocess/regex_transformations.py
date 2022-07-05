"""
Regex-based text transformations for use with the `RegexTransformation` preprocessing
base class.
"""


import dask.dataframe as dd
import re

from spacy.lang.char_classes import (_hyphens, _punct, LIST_QUOTES,
                                     group_chars, merge_chars)

from string import punctuation, whitespace

from sub_clf.preprocess.base import RegexTransformation
from sub_clf.util.utils import pretty_dumps


class ApostropheNormalizer(RegexTransformation):
    """
    Normalize all apostrophes to a single standard form.

    E.g.:

    |Lorem ip‛sum dolor‘s sit amet
        -->
    |Lorem ip'sum dolor's sit amet
    """

    apostrophe_indices = [0, 4, 5, 6, 7]
    apostrophes = [LIST_QUOTES[i] for i in apostrophe_indices]

    pattern = r'|'.join(apostrophes)
    replacement = "'"
    _transformation = {re.compile(pattern) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class CodeBlockRemover(RegexTransformation):
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
    replacement = ''
    _transformation = {re.compile(pattern, re.MULTILINE) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class ConsecutiveNewlineCollapser(RegexTransformation):
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
    replacement = '\n'
    _transformation = {re.compile(pattern, re.MULTILINE) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class HyperlinkRemover(RegexTransformation):
    """
    Remove hyperlinks from comments.

    E.g.:

    |Lorem ipsum dolor sit amet, [consectetur](https://www.website.com) adipiscing elit
        -->
    |Lorem ipsum dolor sit amet, [consectetur]( adipiscing elit
    """

    pattern = r'http\S+'
    replacement = ''
    _transformation = {re.compile(pattern) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class HyphenNormalizer(RegexTransformation):
    """
    Normalize all hyphens, hyphen-like characters, and multi-hyphen sequences to a
    single standard form.

    E.g.:

    |Lorem-ipsum–-dolor---sit
        -->
    |Lorem-ipsum–dolor-sit
    """

    pattern = merge_chars(_hyphens)[:-2] # NB: `:-2` excludes "~" from the normalization
    replacement = '-'
    _transformation = {re.compile(pattern) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class InlineCodeRemover(RegexTransformation):
    """
    Remove inline code (i.e., "`code`") from comments.

    E.g.:

    |Lorem ipsum `dolor` sit amet
        -->
    |Lorem ipsum  sit amet
    """

    pattern = r'`.+?`'
    replacement = ''
    _transformation = {re.compile(pattern) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class PunctuationRemover(RegexTransformation):
    """
    Remove punctuation from comments.

    E.g.:

    |Here's "Lorem-ipsum. dolor sit @met, <<consectetur>> adipiscing elit!"
        -->
    |Here's "Lorem-ipsum dolor sit met consectetur adipiscing elit"
    """

    pattern = merge_chars(_punct)
    replacement = ''
    _transformation = {re.compile(pattern) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class QuotationMarkNormalizer(RegexTransformation):
    """
    Normalize all quotation marks to a single standard form.

    E.g.:

    |『Lorem ipsum』 “dolor‘s” sit "amet"
        -->
    |"Lorem ipsum" "dolor's" sit "amet"
    """

    quotation_mark_indices = [1, 2, 3, *range(8, 27)]
    quotation_marks = [LIST_QUOTES[i] for i in quotation_mark_indices]

    pattern = r'|'.join(quotation_marks)
    replacement = '"'
    _transformation = {re.compile(pattern) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class QuoteRemover(RegexTransformation):
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
    replacement = '\n'
    _transformation = {re.compile(pattern) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)


class WhitespaceNormalizer(RegexTransformation):
    """
    Normalize all whitespace characters to the same form, then collapse sequences of
    consecutive whitespace characters down to a single character.

    E.g.:

    |Lorem   \t\t\n\n\t \t\n\r\x0b\x0c ipsum
        -->
    |Lorem ipsum
    """

    whitespace = group_chars(
        '\u000A \u000B \u000C \u000D \u0009 '
        '\u0020 \u00A0 \u1680 \u180E \u2000 '
        '\u2001 \u2002 \u2003 \u2004 \u2005 '
        '\u2006 \u2007 \u2008 \u2009 \u200A '
        '\u200B \u202F \u205F \u3000 \uFEFF '
    )

    pattern = fr'[{whitespace}]+'
    replacement = ' '
    _transformation = {re.compile(pattern) : replacement}

    def __init__(self):
        RegexTransformation(self._transformation)
