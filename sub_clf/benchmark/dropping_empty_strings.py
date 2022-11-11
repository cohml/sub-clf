"""
Certain preprocessing steps can result in a comment being reduced to an empty
string ''. This may cause problems for downstream feature extractors. So
comments thus affected must be dropped, ideally during preprocessingself.

This script benchmarks the following options for performing this operation:

    *.DataFrame.loc
    *.Series.mask
    *.Series.replace

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from time import perf_counter


def clock(func, *data):
    def wrapper(data):
        start = perf_counter()
        func(data)
        end = perf_counter()
        return end - start
    return wrapper


@clock
def loc(data):
    data.loc[data.text == '', 'text'] = np.nan


@clock
def mask(data):
    data.text.mask(data.text == '', np.nan)


@clock
def replace(data):
    data.text.replace('', np.nan)


def generate_data(n):
    return pd.DataFrame([''], columns=['text']).sample(int(n), replace=True)


def get_n_sizes(max_base=7):
    return np.logspace(0, max_base, 100)


def plot(mean_times, n_sizes):
    ax = plt.subplot(yscale='linear', xlabel='array size', ylabel='mean seconds')

    for method, times in mean_times.items():
        ax.plot(n_sizes, times, label=method)
    ax.legend()

    outpath = Path(__file__).parent / 'dropping_empty_strings.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print('written:', outpath.resolve())


def main(N):
    mean_times = {'loc' : [], 'mask' : [], 'replace' : []}
    n_sizes = get_n_sizes()

    for n in n_sizes:
        print(f'{n = :,.0f}')

        # loc
        print('\tloc', end='\t')
        cumulative_time = 0
        for i in range(N):
            print(i + 1, end=' ', flush=True)
            data = generate_data(n)
            cumulative_time += loc(data)
        mean_time = cumulative_time / N
        mean_times['loc'].append(mean_time)
        print(f'--- {mean_time:.5f} seconds')

        # mask
        print('\tmask', end='\t')
        cumulative_time = 0
        for i in range(N):
            print(i + 1, end=' ', flush=True)
            data = generate_data(n)
            cumulative_time += mask(data)
        mean_time = cumulative_time / N
        mean_times['mask'].append(mean_time)
        print(f'--- {mean_time:.5f} seconds')

        # replace
        print('\treplace', end='\t')
        cumulative_time = 0
        for i in range(N):
            print(i + 1, end=' ', flush=True)
            data = generate_data(n)
            cumulative_time += replace(data)
        mean_time = cumulative_time / N
        mean_times['replace'].append(mean_time)
        print(f'--- {mean_time:.5f} seconds')

    plot(mean_times, n_sizes)


if __name__ == '__main__':
    main(5)
