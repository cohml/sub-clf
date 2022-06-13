"""
Benchmark the performance of two options for parallelized text preprocessing:

    1. using `spacy`'s `nlp.pipe` with custom components
    2. using `sklearn.Pipeline` with custom regex-based transformers

For each option, execute `NITER` iterations, compute the mean processing time,
and save the results to a "benchmark_results.txt" file alongside this script.
"""


import warnings
warnings.simplefilter("ignore", UserWarning)

import dask.dataframe as dd
import pandas as pd
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token

from pathlib import Path
from string import punctuation
from time import perf_counter
from tqdm import tqdm
from typing import Any, Dict, Optional

from sub_clf.util.defaults import DEFAULTS


# number of iterations to execute for computing mean performance benchmarks
NITER = 100

# use data from subreddit with comments guaranteed to contain inline code
TEST_DATA_PATH = DEFAULTS['PATHS']['DIRS']['RAW_DATA'] / 'subreddit=LanguageTechnology'


class PerformanceBenchmarker:
    """Abstract base class for benchmarking performance."""

    def benchmark_performance(self, niter: int, *args, **kwargs):
        """Compute mean duration of data transformation over `niter` iterations."""

        duration = 0
        for _ in tqdm(range(niter), total=niter):
            start = perf_counter()
            self.preprocess(*args, **kwargs)
            end = perf_counter()
            duration += end - start
        return duration / niter

    @staticmethod
    def write_results_to_file(niter: int, durations: Dict[str, float]):
        """Record benchmarking results in a .txt file alongside this script."""

        results_file_path = Path(__file__).parent / 'benchmark_results.txt'
        results_file = results_file_path.open('w')

        print('Mean results over', niter, 'iterations', file=results_file)
        print('================================\n', file=results_file)

        for method, duration in durations.items():
            print(method, '\n\tMean wall time:', duration, 'seconds\n', file=results_file)

        results_file.close()

        print('Results written to', results_file_path.resolve())

    def __str__(self):
        return self.__class__.__name__


# ---- `spacy`'s `nlp.pipe` components ...


class InlineCodeMatcher:
    """Annotate tokens (incl. "`") as belonging or not belonging to spans of inline code."""

    pattern = [{'TEXT' :            '`',   'OP' : '+'},
               {'TEXT' : {'REGEX' : '.+'}, 'OP' : '+'},
               {'TEXT' :            '`',   'OP' : '+'}]

    def __init__(self, vocab, greedy=None):
        Token.set_extension("is_inline_code", default=False)
        self.matcher = Matcher(vocab)
        self.matcher.add('INLINE_CODE', [self.pattern], greedy=greedy)

    def __call__(self, doc):
        inline_code_spans = [doc[i:j] for _, i, j in self.matcher(doc)]
        for span in inline_code_spans:
            for token in span:
                token._.is_inline_code = True
        return doc


class NlpPipe(PerformanceBenchmarker):
    """Wrapper around custom `spacy` pipeline for removing inline code and punctuation."""

    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe('inline_code_matcher')

    @Language.factory("inline_code_matcher")
    def create_inline_code_matcher(nlp, name):
        return InlineCodeMatcher(nlp.vocab)

    def get_annotations_df(self, docs):
        annotations = []
        for doc_id, doc in enumerate(docs):
            for token in doc:
                annotations.append(
                    {
                        'doc_id' : doc_id,
                        'text' : token.text,
                        'is_punct' : token.is_punct,
                        'is_inline_code' : token._.is_inline_code
                     }
                )
        return dd.from_pandas(pd.DataFrame(annotations), npartitions=DEFAULTS['NCPU'])

    def preprocess(self, text: dd.Series):
        docs = self.nlp.pipe(text)
        annotations_df = self.get_annotations_df(docs)
        is_punct_or_inline_code = ~(annotations_df.is_punct | annotations_df.is_inline_code)
        annotations_df = annotations_df[is_punct_or_inline_code]
        docs = annotations_df.groupby('doc_id').text.apply(' '.join)
        return docs


# ---- `sklearn.Pipeline` components ...


class TextTransformer(BaseEstimator, TransformerMixin):
    """Abstract data transformation base class."""

    def fit(self, X: dd.Series, y: Optional[Any] = None):
        return self


class InlineCodeRemover(TextTransformer):
    """Remove inline code (i.e., "`code`") from comments."""

    pattern = r'`.+?`'

    def transform(self, text: dd.Series) -> dd.Series:
        inline_code = {'pat' : re.compile(self.pattern),
                       'repl' : '',
                       'regex' : True}
        return text.str.replace(**inline_code)


class PunctuationRemover(TextTransformer):
    """Remove common punctuation from comments."""

    pattern = fr'[{punctuation}]'

    def transform(self, text: dd.Series) -> dd.Series:
        punct = {'pat' : re.compile(self.pattern),
                 'repl' : '',
                 'regex' : True}
        return text.str.replace(**punct)


class SklearnPipeline(PerformanceBenchmarker):
    """Wrapper around custom `sklearn.Pipeline` for removing inline code and punctuation."""

    def preprocess(self, text: dd.Series):
        self.steps = [('inline_code', InlineCodeRemover()),
                      ('punctuation', PunctuationRemover())]
        self.pipeline = Pipeline(self.steps)
        return self.pipeline.fit_transform(text).compute()


# ---- main benchmarking logic ...


if __name__ == '__main__':

    texts = dd.read_parquet(TEST_DATA_PATH, **DEFAULTS['IO']['READ_PARQUET_KWARGS'])

    durations = {}
    parallelization_methods = [NlpPipe(), SklearnPipeline()]

    for method in parallelization_methods:
        print(f'Measuring {method}...')
        duration = method.benchmark_performance(NITER, texts.text)
        durations[str(method)] = round(duration, 4)

    PerformanceBenchmarker.write_results_to_file(NITER, durations)
