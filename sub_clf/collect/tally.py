"""
Count the number of comments collected by subreddit, as well as the number of NaN
samples overall by column.
"""


import argparse
import dask.dataframe as dd
import pandas as pd

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from sub_clf.util.defaults import DEFAULTS
from sub_clf.util.utils import full_path, measure_duration


@measure_duration
def count(p: Path,
          i: int,
          maxlen: int,
          counts: Dict[str, int],
          nulls: Dict[str, int]
         ) -> Tuple[Dict[str, int], Dict[str, int]]:
    subreddit = p.name.split('=')[1]
    status = f'{i:02} {subreddit} .{"."*(maxlen-len(subreddit))} '
    print(status, end='', flush=True)

    df = dd.read_parquet(p, **DEFAULTS['IO']['READ_PARQUET_KWARGS']).compute()
    counts[subreddit] += len(df)
    for col in nulls:
        nulls[col] += df[col].isna().sum()

    return counts, nulls


def display_counts(counts: Dict[str, int],
                   nulls: Dict[str, int],
                   log: Optional[Path]
                   ) -> None:

    counts = pd.Series(counts)
    n = format(counts.sum(), ',')
    pcts = counts / counts.sum()
    total = (pd.DataFrame({'sub':['------------', 'total'],
                           'n':['----------', n],
                           '%':['-------', '100.00%']})
               .set_index('sub'))

    pd.options.display.max_rows = None

    print('\ndistribution of samples by subreddit:',
          pd.concat([pd.concat([counts.apply('{:,}'.format), pcts], axis=1)
            .rename(columns=dict(enumerate(list('n%'))))
            .sort_values('%', ascending=False)
            .assign(**{'%' : lambda df: df['%'].apply('{:.2%}'.format)}), total]),
          sep='\n',
          file=log)

    nulls = pd.Series(nulls)
    print('\nno. nulls by col:', nulls, sep='\n', file=log)


def get_subreddit_paths(top_level_parquets_dir: Path) -> Tuple[List[Path], int]:
    key = lambda x: str(x).lower()
    parquet_files = sorted(top_level_parquets_dir.iterdir(), key=key)
    maxlen = max(len(p.name.split('=')[1]) for p in parquet_files)
    return parquet_files, maxlen


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--top_level_parquets_dir',
        required=True,
        type=Path
    )
    parser.add_argument(
        '-o', '--output_filepath',
        type=full_path,
        help='log tallies to file, else print to stdout'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log = open(args.output_filepath, 'w') if args.output_filepath else None

    counts = defaultdict(int)
    nulls = dict.fromkeys(['subreddit', 'post_id', 'upvotes', 'text'], 0)

    parquet_files, maxlen = get_subreddit_paths(args.top_level_parquets_dir)
    print('\ncounting samples from', len(parquet_files), 'subreddits:')

    for i, p in enumerate(parquet_files, start=1):
        if not p.is_dir():
            continue

        counts, nulls = count(p, i, maxlen, counts, nulls)

    display_counts(counts, nulls, log)

    if log:
        log.close()
        print('\ntallies written to', args.output_filepath)


if __name__ == '__main__':
    main()
