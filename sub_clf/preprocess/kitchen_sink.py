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

        # standardize certain individual characters (transformations require literals)
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

        # standardize apostrophes (would interfere with `InlineCodeRemover` because ` -> ')
        pp.RegexTransformer(
            transformations=[
                rt.ApostropheNormalizer(),
            ],
            name='ApostropheNormalizer'
        ),

        # remove all remaining punctuation characters
        pp.RegexTransformer(
            transformations=[
                rt.PunctuationRemover()
            ],
            name='PunctuationRemover'
        ),

        # standardize and collapse whitespace (`PunctuationRemover` may result in "  ")
        pp.RegexTransformer(
            transformations=[
                rt.WhitespaceNormalizer()
            ],
            name='WhitespaceNormalizer'
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
