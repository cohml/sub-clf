"""
`sklearn.pipeline.Pipeline` subclass which by default will apply all preprocessors
currently defined in `preprocessers.py`.
"""


from overrides import overrides

from .abstract import MultiplePreprocessorPipeline
from .preprocessors import *


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
