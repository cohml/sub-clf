"""
Benchmark the performance of two options for parallelized text preprocessing:

    1. using `spacy`'s `nlp.pipe` with custom components and a `batch_size` of `NCPU`
    2. using `sklearn.Pipeline` with custom regex-based transformers

For each option, execute `NITER` iterations and compute the mean processing time,
saving (1) the aggregate results to a "benchmark_aggregates.txt" and (2) the raw
results to a series of histograms in "benchmark_histograms.png".
"""


import warnings
warnings.simplefilter("ignore", UserWarning)

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
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

        durations = []
        for _ in tqdm(range(niter), total=niter):
            start = perf_counter()
            self.preprocess(*args, **kwargs)
            end = perf_counter()
            durations.append(end - start)
        return pd.Series(durations).round(4)

    @staticmethod
    def save_results(niter: int, ntext: int, durations: Dict[str, dd.Series]):
        """Save benchmarking results to .txt and .png files."""

        results = pd.DataFrame(durations)
        output_dir = Path(__file__).parent / 'benchmark_results'
        output_dir.mkdir(exist_ok=True, parents=True)

        # save aggregate results (i.e., means and standard deviations) to .txt file
        aggregates_file_path = output_dir / 'benchmark_aggregates.txt'
        aggregates_file = aggregates_file_path.open('w')

        print(f'Aggregate results* ({niter} iterations, {ntext} texts)', file=aggregates_file)
        print('===============================================\n', file=aggregates_file)
        print(results.agg(['mean', 'std']).T.to_string(), file=aggregates_file)
        print('\n* units are in seconds', file=aggregates_file)

        aggregates_file.close()
        print('Results saved to', aggregates_file_path.resolve())

        # save histograms of raw durations to .png file
        min_, max_ = results.stack().agg(['min', 'max'])
        min_, max_ = np.floor(min_), np.ceil(max_)
        nbins = int((max_ - min_) * 4) + 1
        bins = np.linspace(min_, max_, nbins)

        axes = results.plot.hist(bins=bins, title=results.columns.tolist(),
                                 figsize=(10, 6), color='silver',
                                 subplots=True, legend=False)
        axes[-1].set_xticks(bins)
        axes[-1].set_xticklabels('' if i % 1 else int(i) for i in bins)
        axes[-1].set_xlabel('Wall time (sec)')
        plt.suptitle(f'Total duration to preprocess {ntext:,} Reddit commments using '
                     'two different pipelines')

        histogram_file_path = output_dir / 'benchmark_histograms.png'
        plt.savefig(histogram_file_path, dpi=150)
        print('Histograms saved to', histogram_file_path.resolve())


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
        docs = self.nlp.pipe(text, batch_size=DEFAULTS['NCPU'])
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

    # read in reddit comment data as dask.dataframe (raw text in "text" column)
    texts = dd.read_parquet(TEST_DATA_PATH, **DEFAULTS['IO']['READ_PARQUET_KWARGS'])

    # initialize pipelines to benchmark and container for results
    parallelization_methods = [NlpPipe(), SklearnPipeline()]
    durations = {}

    # perform `NITER` iterations for each pipelne and capture results
    for method in parallelization_methods:
        print(f'Benchmarking {method}...')
        duration = method.benchmark_performance(NITER, texts.text)
        durations[str(method)] = duration

    # write aggregate results to .txt file and plot histograms as .png
    PerformanceBenchmarker.save_results(NITER, len(texts), durations)
