"""
`sklearn.pipeline.Pipeline` subclass which by default will apply all preprocessors
currently defined in `preprocessers.py`.
"""


from overrides import overrides

from preprocessing.abstract import MultiplePreprocessorPipeline
from preprocessing.preprocessors import *


class KitchenSinkPreprocessor(MultiplePreprocessorPipeline):
    """Apply all available preprocesing steps to comments."""

    preprocessors = [CaseNormalizer(),
                     HyperlinkRemover(),
                     InlineCodeRemover(),
                     CodeBlockRemover(),
                     QuoteRemover(),
                     AccentRemover(),
                     ApostropheNormalizer(),
                     PunctuationRemover(),
                     WhitespaceNormalizer(),
                     NewlineCollapser(),
                     Stemmer()]


    @overrides
    def __init__(self, verbose: bool = True):
        super().__init__(*self.preprocessors, verbose=verbose)
