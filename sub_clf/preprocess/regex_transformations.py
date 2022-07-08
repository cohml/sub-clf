"""
Regex-based text transformations for use with the `RegexTransformation` preprocessing
base class.
"""


import dask.dataframe as dd
import re

from spacy.lang.char_classes import _hyphens, LIST_QUOTES, merge_chars

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
    apostrophes += ['‛']

    pattern = r'|'.join(apostrophes)
    replacement = "'"
    _transformations = [(re.compile(pattern), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)


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
    _transformations = [(re.compile(pattern, re.MULTILINE), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)


class HTMLConverter(RegexTransformation):
    """
    Convert HTML character codes to literal characters. Converts only the small subset
    of codes which seem relevant.

    E.g.:

    |&amp;&#32;Lorem&#32;ipsum&#32;dolor&#32;sit&#32;amet
        -->
    |> Lorem ipsum dolor sit amet
    """

    _transformations = [
        (re.compile(r'&#32;'), ' '),
        (re.compile(r'&#38;'), '&'), (re.compile(r'&amp;'), '&'),
        (re.compile(r'&#60;'), '<'), (re.compile(r'&lt;'), '<'),
        (re.compile(r'&#62;'), '>'), (re.compile(r'&gt;'), '>'),
        (re.compile(r'&#160;'), ' '), (re.compile(r'&nbsp;'), ' '),
        (re.compile(r'&#732;'), '˜'), (re.compile(r'&tilde;'), '˜'),
        (re.compile(r'&#8194;'), '\u2002'), (re.compile(r'&ensp;'), '\u2002'),
        (re.compile(r'&#8195;'), '\u2003'), (re.compile(r'&emsp;'), '\u2003'),
        (re.compile(r'&#8201;'), '\u2009'), (re.compile(r'&thinsp;'), '\u2009'),
        (re.compile(r'&#8204;'), '\u200C'), (re.compile(r'&zwnj;'), '\u200C'),
        (re.compile(r'&#8205;'), '\u200D'), (re.compile(r'&zwj;'), '\u200D'),
        (re.compile(r'&#8206;'), '\u200E'), (re.compile(r'&lrm;'), '\u200E'),
        (re.compile(r'&#8207;'), '\u200F'), (re.compile(r'&rlm;'), '\u200F'),
        (re.compile(r'&#8211;'), '–'), (re.compile(r'&ndash;'), '–'),
        (re.compile(r'&#8212;'), '—'), (re.compile(r'&mdash;'), '—'),
        (re.compile(r'&#8216;'), '‘'), (re.compile(r'&lsquo;'), '‘'),
        (re.compile(r'&#8217;'), '’'), (re.compile(r'&rsquo;'), '’'),
        (re.compile(r'&#8218;'), '‚'), (re.compile(r'&sbquo;'), '‚'),
        (re.compile(r'&#8220;'), '“'), (re.compile(r'&ldquo;'), '“'),
        (re.compile(r'&#8221;'), '”'), (re.compile(r'&rdquo;'), '”'),
        (re.compile(r'&#8222;'), '„'), (re.compile(r'&bdquo;'), '„'),
        (re.compile(r'&#8226;'), '•'), (re.compile(r'&bull;'), '•'),
        (re.compile(r'&#8230;'), '…'), (re.compile(r'&hellip;'), '…'),
        (re.compile(r'&#8242;'), '′'), (re.compile(r'&prime;'), '′'),
        (re.compile(r'&#8243;'), '″'), (re.compile(r'&Prime;'), '″'),
        (re.compile(r'&#8249;'), '‹'), (re.compile(r'&lsaquo;'), '‹'),
        (re.compile(r'&#8250;'), '›'), (re.compile(r'&rsaquo;'), '›')
    ]


    def __init__(self):
        RegexTransformation(self._transformations)


class HyperlinkRemover(RegexTransformation):
    """
    Remove hyperlinks and URLs from comments.

    E.g.:

    |Lorem ipsum dolor sit amet, [consectetur](https://www.website.com) adipiscing elit
        -->
    |Lorem ipsum dolor sit amet, [consectetur]( adipiscing elit
    """

    pattern = r'(http|www)\S+'
    replacement = ''
    _transformations = [(re.compile(pattern), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)


class HyphenNormalizer(RegexTransformation):
    """
    Normalize all hyphens, hyphen-like characters (excluding ~), and multi-hyphen
    sequences to a single standard form.

    E.g.:

    |~Lorem-ipsum–dolor—sit---  # these hyphens look identical but are all different characters
        -->
    |~Lorem-ipsum–dolor-sit-
    """

    hyphens = merge_chars(_hyphens[:-1])  # exclude "~" from normalization

    pattern = fr'({hyphens})+'
    replacement = '-'
    _transformations = [(re.compile(pattern), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)


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
    _transformations = [(re.compile(pattern), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)


class PunctuationRemover(RegexTransformation):
    """
    Remove punctuation from comments. Apostrophes, hyphens, slashes, and underscores
    are left alone, lest their removal complicate downstream tokenization.

    E.g.:

    |Here's "Lorem-ipsum. dolor sit @met, <<consectetur>> adipiscing elit!"
        -->
    |Here's "Lorem-ipsum dolor sit met consectetur adipiscing elit"
    """

    pattern = r'[^\w\s\'\-/_]'
    replacement = ''
    _transformations = [(re.compile(pattern), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)


class QuotationMarkNormalizer(RegexTransformation):
    """
    Normalize all quotation marks to a single standard form.

    E.g.:

    |『Lorem ipsum』 “dolor‘s” sit "amet"
        -->
    |"Lorem ipsum" "dolor's" sit "amet"
    """

    quotation_mark_indices = [1, 2, 3, *range(10, 27)]
    quotation_marks = [LIST_QUOTES[i] for i in quotation_mark_indices]

    pattern = r'|'.join(quotation_marks)
    replacement = '"'
    _transformations = [(re.compile(pattern), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)


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
    |ut labore et dolore magna aliqua.
    """

    pattern = r'(^|\n) {,3}(&gt;|>).*?\n'    # NB: ">" is sometimes rendered as "&gt;"
    replacement = ''
    _transformations = [(re.compile(pattern), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)


class WhitespaceNormalizer(RegexTransformation):
    """
    Normalize all whitespace characters to the same form, then collapse sequences of
    consecutive whitespace characters down to a single character.

    E.g.:

    |Lorem   \t\t\n\n\t \t\n\r\x0b\x0c \u000A \u000B \u000C ipsum
        -->
    |Lorem ipsum
    """

    whitespace = merge_chars(
        '\u000A \u000B \u000C \u000D \u0009 '
        '\u0020 \u00A0 \u1680 \u180E \u2000 '
        '\u2001 \u2002 \u2003 \u2004 \u2005 '
        '\u2006 \u2007 \u2008 \u2009 \u200A '
        '\u200B \u202F \u205F \u3000 \uFEFF '
        '\n \t \v \b \r \f \a'
    )
    whitespace += '| '

    pattern = fr'({whitespace})+'
    replacement = ' '
    _transformations = [(re.compile(pattern, re.MULTILINE), replacement)]


    def __init__(self):
        RegexTransformation(self._transformations)
