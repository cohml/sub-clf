"""
Load all objects available for use in this package.
"""

## ---- feature extractors
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer
)
from embed.embeddings import LgEmbeddingsVectorizer, TrfEmbeddingsVectorizer

## ---- preprocessors
from preprocess.kitchen_sink import KitchenSinkPreprocessor
from preprocess.preprocessors import (
    AccentRemover,
    ApostropheNormalizer,
    CaseNormalizer,
    CodeBlockRemover,
    HyperlinkRemover,
    InlineCodeRemover,
    NewlineCollapser,
    PunctuationRemover,
    QuoteRemover,
    SinglePreprocessor,
    Stemmer,
    WhitespaceNormalizer
)


## ---- sklearn models
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LinearRegression,
    LogisticRegression,
    RANSACRegressor,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

## ---- spacy models
import en_core_web_lg
import en_core_web_trf


AVAILABLE = {
    'FEATURE_EXTRACTORS' : {
        'HashingVectorizer' : HashingVectorizer,
        'CountVectorizer' : CountVectorizer,
        'LgEmbeddingsVectorizer' : LgEmbeddingsVectorizer,
        'TfidfTransformer' : TfidfTransformer,
        'TfidfVectorizer' : TfidfVectorizer,
        'TrfEmbeddingsVectorizer' : TrfEmbeddingsVectorizer
    },
    'PREPROCESSING' : {
        'PREPROCESSORS' : {
            'AccentRemover' : AccentRemover,
            'ApostropheNormalizer': ApostropheNormalizer,
            'CaseNormalizer': CaseNormalizer,
            'CodeBlockRemover': CodeBlockRemover,
            'HyperlinkRemover': HyperlinkRemover,
            'InlineCodeRemover': InlineCodeRemover,
            'NewlineCollapser': NewlineCollapser,
            'PunctuationRemover': PunctuationRemover,
            'QuoteRemover': QuoteRemover,
            'SinglePreprocessor': SinglePreprocessor,
            'Stemmer': Stemmer,
            'WhitespaceNormalizer': WhitespaceNormalizer
        },
        'PIPELINES' : {
            'KitchenSinkPreprocessor' : KitchenSinkPreprocessor
        }
    },
    'MODELS' : {
        'PYTORCH' : None,
        'SKLEARN' : {
            'AdaBoostClassifier': AdaBoostClassifier,
            'AdaBoostRegressor': AdaBoostRegressor,
            'BayesianRidge': BayesianRidge,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'ElasticNet': ElasticNet,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'HuberRegressor': HuberRegressor,
            'KNeighborsClassifier': KNeighborsClassifier,
            'KNeighborsRegressor': KNeighborsRegressor,
            'Lars': Lars, 'Lasso': Lasso,
            'LinearRegression': LinearRegression,
            'LinearSVC': LinearSVC,
            'LinearSVR': LinearSVR,
            'LogisticRegression': LogisticRegression,
            'MLPClassifier': MLPClassifier,
            'MLPRegressor': MLPRegressor,
            'MultinomialNB': MultinomialNB,
            'RANSACRegressor': RANSACRegressor,
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForestRegressor': RandomForestRegressor,
            'Ridge': Ridge,
            'RidgeClassifier': RidgeClassifier,
            'SGDClassifier': SGDClassifier,
            'SGDRegressor': SGDRegressor,
            'SVC': SVC,
            'SVR': SVR,
            'TheilSenRegressor': TheilSenRegressor
        },
        'SPACY' : {
            'Lg' : lambda: en_core_web_lg.load(),
            'Trf' : lambda: en_core_web_trf.load()
        }
    }
}
