"""
Benchmark the performance various parallelized text preprocessing pipelines. Details
will vary based on the argument passed.

If you pass "kitchen_sink", then the `KitchenSinkPreprocessor` will be benchmarked
using the the largest subreddit..

If you pass "sklearn_vs_spacy", then two pipelines will be comparatively benchmarked:

    1. using `spacy`'s `nlp.pipe` with custom components and a `batch_size` of `NCPU`
    2. using `sklearn.Pipeline` with custom regex-based transformers

For each option, execute `niter` iterations (varies by option; defined in the
`BENCHMARKING_PARAMETERS` dictionary below) and compute the mean processing time,
saving (1) the aggregate results to a "benchmark_aggregates.md" and (2) the raw
results to a series of histograms in "benchmark_histograms.png".
"""


import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
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

from sub_clf.preprocess.kitchen_sink import KitchenSinkPreprocessor
from sub_clf.util.defaults import DEFAULTS


class PerformanceBenchmarker:
    """Abstract base class for benchmarking performance."""

    def benchmark_performance(
        self,
        what_to_benchmark: str,
        niter: int,
        *args,
        **kwargs
    ) -> pd.Series:
        """Compute mean duration of data transformation over `niter` iterations."""

        durations = []
        if what_to_benchmark == 'kitchen_sink':
            iterations = range(1, niter + 1)
            print_status = True
        elif what_to_benchmark == 'sklearn_vs_spacy':
            iterations = tqdm(range(niter), total=niter)
            print_status = False

        for i in iterations:
            if print_status: print(f'Iteration {i:0{len(str(niter))}}/{niter}:', end='\t', flush=True)
            start_all = perf_counter()
            result = self.preprocess(*args, **kwargs)
            end_preprocess = perf_counter()
            duration_preprocess = end_preprocess - start_all
            if print_status: print('preprocess', round(duration_preprocess / 60, 1), end='\t', flush=True)
            if hasattr(result, 'compute'):
                result.compute()
            end_all = perf_counter()
            duration_compute = end_all - end_preprocess
            if print_status: print('compute', round(duration_compute / 60, 1), end='\t', flush=True)
            duration_compute = end_all - end_preprocess
            duration_all = end_all - start_all
            if print_status: print('all', round(duration_all / 60, 1), '\t(minutes)')
            durations.append(duration_all)

        return pd.Series(durations).round(4)


    @staticmethod
    def save_results(
        what_to_benchmark: str,
        niter: int,
        ntext: int,
        durations: Dict[str, dd.Series]
    ):
        """Save benchmarking results to .md and .png files."""

        if what_to_benchmark == 'kitchen_sink':
            outdir_suffix = 'ksp'
        elif what_to_benchmark == 'sklearn_vs_spacy':
            outdir_suffix = 'sk_v_sp'

        results = pd.DataFrame(durations)
        output_dir = Path(__file__).parent / f'preprocessing_pipelines_results_{outdir_suffix}'
        output_dir.mkdir(exist_ok=True, parents=True)

        # save aggregate results (i.e., means and standard deviations) to .md file
        aggregates_file_path = output_dir / 'benchmark_aggregates.md'
        aggregates_file = aggregates_file_path.open('w')

        header = f'Aggregate results ({niter} iterations, {ntext:,} texts)'
        print(header, file=aggregates_file)
        print('=' * len(header) + '\n', file=aggregates_file)
        print(results.agg(['mean', 'std']).T.to_markdown(tablefmt='pipe'), file=aggregates_file)
        print('\n> ℹ️  All units are in seconds', file=aggregates_file)

        aggregates_file.close()
        print('Results saved to', aggregates_file_path.resolve())

        # save histograms of raw durations to .png file
        min_, max_ = results.stack().agg(['min', 'max'])
        min_, max_ = np.floor(min_), np.ceil(max_)
        nbins = niter if what_to_benchmark == 'kitchen_sink' else int((max_ - min_) * 4) + 1
        bins = np.linspace(min_, max_, nbins)

        axes = results.plot.hist(bins=bins, title=results.columns.tolist(),
                                 figsize=(10, 6), color='silver',
                                 subplots=True, legend=False)
        axes[-1].set_xticks(bins)
        axes[-1].set_xticklabels('' if i % 1 else int(i) for i in bins)
        axes[-1].set_xlabel('Wall time (sec)')
        plt.suptitle(f'Total duration to preprocess {ntext:,} Reddit commments')

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
        Token.set_extension('is_inline_code', default=False)
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

    @Language.factory('inline_code_matcher')
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

    def preprocess(self, data: dd.core.DataFrame):
        docs = self.nlp.pipe(data.text, batch_size=DEFAULTS['NCPU'])
        annotations_df = self.get_annotations_df(docs)
        is_punct_or_inline_code = annotations_df.is_punct | annotations_df.is_inline_code
        annotations_df = annotations_df[~is_punct_or_inline_code]
        docs = annotations_df.groupby('doc_id').text.apply(' '.join)
        return docs


# ---- `sklearn.Pipeline` components ...


class TextTransformer(BaseEstimator, TransformerMixin):
    """Abstract data transformation base class."""

    def fit(self, X: dd.core.DataFrame, y: Optional[Any] = None):
        return self


class InlineCodeRemover(TextTransformer):
    """Remove inline code (i.e., "`code`") from comments."""

    pattern = r'`.+?`'

    def transform(self, data: dd.core.DataFrame) -> dd.core.DataFrame:
        inline_code = {'pat' : re.compile(self.pattern),
                       'repl' : '',
                       'regex' : True}
        data.text = data.text.str.replace(**inline_code)
        return data


class PunctuationRemover(TextTransformer):
    """Remove common punctuation from comments."""

    pattern = fr'[{punctuation}]'

    def transform(self, data: dd.core.DataFrame) -> dd.core.DataFrame:
        punct = {'pat' : re.compile(self.pattern),
                 'repl' : '',
                 'regex' : True}
        data.text = data.text.str.replace(**punct)
        return data


class SklearnPipeline(PerformanceBenchmarker):
    """Wrapper around custom `sklearn.Pipeline` for removing inline code and punctuation."""

    def preprocess(self, data: dd.core.DataFrame):
        self.steps = [('inline_code', InlineCodeRemover()),
                      ('punctuation', PunctuationRemover())]
        self.pipeline = Pipeline(self.steps)
        return self.pipeline.fit_transform(data).compute()


# ---- KitchenSinkProcessor + `benchmark` method


class KitchenSinkPreprocessorPlus(KitchenSinkPreprocessor, PerformanceBenchmarker):
    pass


# ---- main logic ...


BENCHMARKING_PARAMETERS = {
    'kitchen_sink' : {
        'niter' : 5,    # number of iterations to execute for computing mean performance benchmarks
        'pipeline_kwargs' : {'verbose' : False},
        'pipelines' : [KitchenSinkPreprocessorPlus],     # pipeline(s) to benchmark
        'test_data_path' : DEFAULTS['PATHS']['DIRS']['RAW_DATA'] / 'subreddit=AskReddit'    # data subset to use for benchmarking
    },
    'sklearn_vs_spacy' : {
        'niter' : 100,
        'pipeline_kwargs' : {},
        'pipelines' : [NlpPipe, SklearnPipeline],
        'test_data_path' : DEFAULTS['PATHS']['DIRS']['RAW_DATA'] / 'subreddit=LanguageTechnology'
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'what_to_benchmark',
        help='what to benchmark; if "kitchen_sink", largest subreddit wil be used; '
             'if "sklearn_vs_spacy", a smaller data subset will be used '
             '(default: %(default)s)',
        default='sklearn_vs_spacy',
        choices=BENCHMARKING_PARAMETERS.keys()
    )
    args = parser.parse_args()

    # set benchmarking parameters according to requested pipeline(s)
    params = BENCHMARKING_PARAMETERS[args.what_to_benchmark]
    niter = params['niter']
    pipeline_kwargs = params['pipeline_kwargs']
    pipelines = [pipeline(**pipeline_kwargs) for pipeline in params['pipelines']]
    test_data_path = params['test_data_path']

    # read in reddit comment data as dask.dataframe (raw text in "text" column)
    data = dd.read_parquet(test_data_path, **DEFAULTS['IO']['READ_PARQUET_KWARGS'])
    print(f'Benchmarking on {test_data_path.name} ({len(data):,} comments)')

    # perform `niter` iterations for each pipelne and capture results
    durations = {}
    for pipeline in pipelines:
        print(f'Benchmarking {pipeline}...')
        duration = pipeline.benchmark_performance(args.what_to_benchmark, niter, data)
        durations[str(pipeline)] = duration

    # write aggregate results to .md file and plot histograms as .png
    PerformanceBenchmarker.save_results(args.what_to_benchmark, niter, len(data), durations)


if __name__ == '__main__':
    main()
