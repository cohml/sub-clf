"""
`sklearn.pipeline.Pipeline` subclass which by default will apply all preprocessors
currently defined in `preprocessers.py`.
"""


from overrides import overrides

from sub_clf.preprocess.base import MultiplePreprocessorPipeline
from sub_clf.preprocess.preprocessors import *


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
                     StopwordRemover(lemmatize=True)]


    @overrides
    def __init__(self, verbose: bool = True):
        super().__init__(*self.preprocessors, verbose=verbose)
