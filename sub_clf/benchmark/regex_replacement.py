"""
Benchmark the performance of regex replacement on `dask.Series` when performing
single replacements in series via individual function calls versus versus
multiple replacements in parallel via a single function call. Plot and display
results interactively.
"""


import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
import string
import time


def display_status(ni, i, replacement_type):
    array_size = f'array size: {ni:>9,}'
    iter_num = f'iteration: {i + 1}'
    replacement_type = f'replacement type: {replacement_type}'
    print(array_size, iter_num, replacement_type, sep='\t')


def get_dicts():
    singles = [{x : x.upper()} for x in 'abcde']
    multiple = {k : v for d in singles for k, v in d.items()}
    return *singles, multiple


def get_base_s():
    return pd.Series(list(string.ascii_lowercase))


def get_iters():
    return np.logspace(1, 6, 100, endpoint=True, dtype=int)


def get_times(s, n, *dicts):
    d1, d2, d3, d4, d5, d = dicts
    times_single = []
    times_multiple = []

    for ni in n:
        s_ni = s.sample(n=ni, replace=True, random_state=42)
        s_ni_dd = dd.from_pandas(s_ni, npartitions=12)

        # single
        times_single_tmp = []
        for i in range(5):
            display_status(ni, i, 'single')
            start = time.perf_counter()
            (s_ni_dd.replace(d1, regex=True)
                    .replace(d2, regex=True)
                    .replace(d3, regex=True)
                    .replace(d4, regex=True)
                    .replace(d5, regex=True)
                    .compute())
            end = time.perf_counter()
            times_single_tmp.append(end - start)
        times_single_mean = sum(times_single_tmp) / len(times_single_tmp)
        times_single.append(times_single_mean)

        # multiple
        times_multiple_tmp = []
        for i in range(5):
            display_status(ni, i, 'multiple')
            start = time.perf_counter()
            s_ni_dd.replace(d, regex=True).compute()
            end = time.perf_counter()
            times_multiple_tmp.append(end - start)
        times_multiple_mean = sum(times_multiple_tmp) / len(times_multiple_tmp)
        times_multiple.append(times_multiple_mean)

    times = pd.DataFrame({'single' : times_single, 'multiple' : times_multiple})
    times.index = n

    return times


def plot_times(times, n):
    times = times.rolling(3).mean()
    times['delta'] = times['multiple'] - times['single']

    single_is_faster = times['delta'] > 0
    multiple_is_faster = times['delta'] < 0

    fig, axes = plt.subplots(2, figsize=(10, 6), sharex=True)
    times[['single', 'multiple']].plot(ax=axes[0], ylabel='wall time (s)')
    axes[1].fill_between(n, times['delta'], where=single_is_faster, interpolate=True, facecolor='red')
    axes[1].fill_between(n, times['delta'], where=multiple_is_faster, interpolate=True, facecolor='green')
    axes[1].axhline(y=0, color='black', linewidth=0.75, linestyle='--')
    axes[1].set_ylabel('delta (multiple - single) (s)')

    fig.align_labels()
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(n.min(), n.max())
        ax.set_xlabel('array size')
        ax.grid()

    script = Path(__file__).resolve()
    outpath = script.parent / (script.stem + '_results.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print('written:', outpath)


def main():
    d1, d2, d3, d4, d5, d = get_dicts()
    s = get_base_s()
    n = get_iters()
    times = get_times(s, n, d1, d2, d3, d4, d5, d)
    plot_times(times, n)


if __name__ == '__main__':
    main()
