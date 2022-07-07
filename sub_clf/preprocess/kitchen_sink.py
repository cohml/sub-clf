"""
`sklearn.pipeline.Pipeline` subclass which applies all transformations currently
defined in the `preprocessers.py` and `regex_transformations.py` modules.
"""


from overrides import overrides

from sub_clf.preprocess import preprocessors as pp
from sub_clf.preprocess import regex_transformations as rt
from sub_clf.preprocess.base import MultiplePreprocessorPipeline


class KitchenSinkPreprocessor(MultiplePreprocessorPipeline):
    """
    Apply all available preprocesing steps to comments.

    Note: Multiple `pp.RegexTransformer` instances are instantiated because while each
    instance's regex transformations are applied in parallel, some transformations must
    occur before others lest they conflict. Therefore, splitting transformations across
    multiple instances balances the desired parallelism with the required sequentialism.
    """

    regex_transformers = [

        # convert HTML codes to literal characters
        pp.RegexTransformer(
            transformations=[
                rt.HTMLConverter(),
            ],
            name='HTMLConverter'
        ),

        # standardize certain individual characters
        pp.RegexTransformer(
            transformations=[
                rt.HyphenNormalizer(),
                rt.QuotationMarkNormalizer(),
            ],
            name='HyphenQuotesStandardizer'
        ),

        # remove spans of unwanted characters
        pp.RegexTransformer(
            transformations=[
                rt.CodeBlockRemover(),
                rt.HyperlinkRemover(),
                rt.InlineCodeRemover(),
                rt.QuoteRemover()
            ],
            name='GarbageSpanRemover'
        ),

        # standardize other individual characters
        pp.RegexTransformer(
            transformations=[
                rt.ApostropheNormalizer(),
                rt.WhitespaceNormalizer()
            ],
            name='ApostropheWhitespaceStandardizer'
        ),

        # remove all remaining punctuation characters
        pp.RegexTransformer(
            transformations=[
                rt.PunctuationRemover()
            ],
            name='PunctuationRemover'
        )

    ]

    preprocessors = [
        pp.CaseNormalizer(),
        pp.AccentRemover(),
        *regex_transformers,
        pp.StopwordRemover(lemmatize=True)
    ]


    @overrides
    def __init__(self, verbose: bool = True):
        super().__init__(*self.preprocessors, verbose=verbose)
