from pathlib import Path
from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).resolve().parent

setup(
    name='subreddit_comment_classification',
    version='0.0.1',
    description='Resources for building Reddit comment classification models',
    long_description=(PROJECT_ROOT / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/cohml/subreddit_comment_classification',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=['dask>=2022.4.1',
                      'dask-ml>=2022.1.22',
                      'ipython>=8.2.0',
                      'jupyter>=1.0.0',
                      'matplotlib>=3.5.1',
                      'numpy>=1.22.3',
                      'overrides>=6.1.0',
                      'pandas>=1.4.2',
                      'plotly>=5.7.0',
                      'praw>=7.5.0',
                      'pytest>=7.1.2',
                      'scikit-learn>=1.0.2',
                      'spacy>=3.2.4',
                      'torch>=1.11.0',
                      'tqdm>=4.64.0'],
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.10'
    ],
    entry_points={
        'console_scripts':
            [
                'clean = subreddit_comment_classification.scrape.drop_invalid:main',
                'scrape = subreddit_comment_classification.scrape.scrape_data:main'
            ]
        },
)
