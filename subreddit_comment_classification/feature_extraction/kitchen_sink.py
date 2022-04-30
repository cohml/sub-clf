"""
`sklearn.pipeline.Pipeline` subclass which by default will apply all preprocessors
currently defined in `preprocessers.py`.
"""


from overrides import overrides

from ..feature_extraction.abstract import MultiplePreprocessorPipeline
from ..feature_extraction.preprocessors import *


class KitchenSinkPreprocessor(MultiplePreprocessorPipeline):
    """Apply all available preprocesing steps to comments."""

    preprocessors = [CaseNormalizer(),
                     HyperlinkRemover(),
                     InlineCodeRemover(),
                     CodeBlockRemover(),
                     QuoteRemover(),
                     ApostropheNormalizer(),
                     PunctuationRemover(),
                     WhitespaceNormalizer(),
                     NewlineCollapser(),
                     Stemmer()]


    @overrides
    def __init__(self, **pipeline_kwargs):
        super().__init__(*self.preprocessors, **pipeline_kwargs)
