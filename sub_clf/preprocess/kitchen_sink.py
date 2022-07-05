"""
`sklearn.pipeline.Pipeline` subclass which applies all transformations currently
defined in the `preprocessers.py` and `regex_transformations.py` modules.
"""


from overrides import overrides

from sub_clf.preprocess import preprocessors as pp
from sub_clf.preprocess import regex_transformations as rt
from sub_clf.preprocess.base import MultiplePreprocessorPipeline


class KitchenSinkPreprocessor(MultiplePreprocessorPipeline):
    """Apply all available preprocesing steps to comments."""

    regex_transformations = [
        rt.HyperlinkRemover(),
        rt.InlineCodeRemover(),
        rt.CodeBlockRemover(),
        rt.QuoteRemover(),
        rt.HyphenNormalizer(),
        rt.ApostropheNormalizer(),
        rt.QuotationMarkNormalizer(),
        rt.PunctuationRemover(),
        rt.WhitespaceNormalizer(),
        rt.ConsecutiveNewlineCollapser()
    ]

    preprocessors = [
        pp.CaseNormalizer(),
        pp.AccentRemover(),
        pp.RegexTransformer(regex_transformations),
        pp.StopwordRemover(lemmatize=True)
    ]


    @overrides
    def __init__(self, verbose: bool = True):
        super().__init__(*self.preprocessors, verbose=verbose)
