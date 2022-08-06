import dask.dataframe as dd
import pandas as pd
import pytest
import re

from typing import Tuple
from unittest.mock import Mock

from sub_clf.preprocess.base import RegexTransformation, SinglePreprocessor
from sub_clf.preprocess.kitchen_sink import KitchenSinkPreprocessor
from sub_clf.preprocess import (
    preprocessors as pp,
    regex_transformations as rt
)


@pytest.mark.preprocess
@pytest.mark.regex
@pytest.mark.parametrize(
    'invalid_transformation', (
        [(r'.', '.')],
        [(re.compile(r'.'), None)]
    ),
    ids=['pattern', 'replacement']
)
def test_regex_transformation_base_invalid(invalid_transformation):
    """
    Test that the `RegexTransformation` base class' dtype requirements are enforced for
    regex transformations.
    """

    # mock `RegexTransformation`
    regex_transformation = Mock()

    expected_error_message = 'attribute of a `RegexTransformation` subclass must be a'

    with pytest.raises(TypeError, match=expected_error_message):
        RegexTransformation(invalid_transformation)


REGEX_TRANSFORMATION_PARAMETERS = {
    'ApostropheNormalizer' : (
        rt.ApostropheNormalizer,
        r"Lorem ip‛sum dolor‘s sit amet",
        "Lorem ip'sum dolor's sit amet"
    ),
    'CodeBlockRemover' : (
        rt.CodeBlockRemover,
        (
            "Lorem ipsum dolor sit amet,\n"
            "\n"
            "    consectetur adipiscing elit,\n"
            "    sed do eiusmod tempor incididunt\n"
            "\n"
            "ut labore et dolore magna aliqua."
        ),
        (
            "Lorem ipsum dolor sit amet,\n"
            "\n"
            "\n"
            "ut labore et dolore magna aliqua."
        )
    ),
    'HTMLConverter' : (
        rt.HTMLConverter,
        "&gt;&#32;Lorem&#32;ipsum&#32;dolor&#32;sit&#32;amet",
        "> Lorem ipsum dolor sit amet"
    ),
    'HyperlinkRemover' : (
        rt.HyperlinkRemover,
        r"Lorem ipsum dolor sit amet, [consectetur](https://www.website.com) adipiscing elit",
        "Lorem ipsum dolor sit amet, [consectetur]( adipiscing elit"
    ),
    'HyphenNormalizer' : (
        rt.HyphenNormalizer,
        r"~Lorem-ipsum–dolor—sit---",  # these hyphens look identical but are all different characters
        "~Lorem-ipsum-dolor-sit-"
    ),
    'InlineCodeRemover' : (
        rt.InlineCodeRemover,
        r"Lorem ipsum `dolor` sit amet",
        "Lorem ipsum  sit amet"
    ),
    'PunctuationRemover' : (
        rt.PunctuationRemover,
        'Here\'s "Lorem-ipsum. dolor/sit @m3t, <<consectetur>> adipiscing_elit!"',
        'Here\'s  Lorem-ipsum  dolor/sit  m3t    consectetur   adipiscing_elit  '
    ),
    'QuotationMarkNormalizer' : (
        rt.QuotationMarkNormalizer,
        r'『Lorem ipsum』 “dolor‘s” sit "amet"',
        '"Lorem ipsum" "dolor‘s" sit "amet"'
    ),
    'QuoteRemover' : (
        rt.QuoteRemover,
        (
            "Lorem ipsum dolor sit amet,\n"
            "\n"
            "> consectetur adipiscing elit,\n"
            "\n"
            ">> sed do eiusmod tempor incididunt\n"
            "\n"
            "ut labore et dolore magna aliqua."
        ),
        (
            "Lorem ipsum dolor sit amet,\n"
            "\n"
            "ut labore et dolore magna aliqua."
        )
    ),
    'WhitespaceNormalizer' : (
        rt.WhitespaceNormalizer,
        "Lorem   \t\t\n\n\t \t\n\r\x0b\x0c \u000A \u000B \u000C ipsum",
        "Lorem ipsum"
    )
}
@pytest.mark.preprocess
@pytest.mark.regex
@pytest.mark.parametrize(
    'regex_transformation',
    REGEX_TRANSFORMATION_PARAMETERS.values(),
    ids=REGEX_TRANSFORMATION_PARAMETERS.keys()
)
def test_regex_transformation_subclass(regex_transformation: Tuple[RegexTransformation, str, str]):
    """Test the `RegexTransformation` subclasses in `preprocess.regex_transformations`."""

    regex_transformation_obj, raw_input, expected_output = regex_transformation
    computed_output = raw_input

    for pattern, replacement in regex_transformation_obj().transformations:
        computed_output = pattern.sub(replacement, computed_output)

    assert computed_output == expected_output


SIMPLE_PREPROCESSOR_PARAMETERS = {
    'AccentRemover' : (
        pp.AccentRemover,
        "ìíîïñòóôõöùúûüý and \xec\xed\xee\xef\xf1\xf2\xf3\xf4\xf5\xf6\xf9\xfa\xfb\xfc\xfd",
        "iiiinooooouuuuy and iiiinooooouuuuy"
    ),
    'CaseNormalizer' : (
        pp.CaseNormalizer,
        "Lorem ipsum DoloR sit amet.",
        "lorem ipsum dolor sit amet."
    ),
    'PassthroughPreprocessor' : (
        pp.PassthroughPreprocessor,
        "Lorem ipsum dolor sit amet.",
        "Lorem ipsum dolor sit amet."
    )
}
@pytest.mark.preprocess
@pytest.mark.parametrize(
    'simple_preprocessor',
    SIMPLE_PREPROCESSOR_PARAMETERS.values(),
    ids=SIMPLE_PREPROCESSOR_PARAMETERS.keys()
)
def test_simple_preprocessor(simple_preprocessor: Tuple[SinglePreprocessor, str, str]):
    """Test the simpler preprocessor objects in `preprocess.preprocessors`."""

    simple_preprocessor_obj, raw_input, expected_output = simple_preprocessor
    raw_input = pd.DataFrame([raw_input], columns=['text'])

    computed_output = simple_preprocessor_obj().transform(raw_input)
    computed_output = computed_output.text.item()

    assert computed_output == expected_output


COMPLEX_PREPROCESSOR_PARAMETERS = {
    'Stemmer' : (
        pp.Stemmer,
        'a highly inflected sentence is one having lots of morphemes for stemming',
        'a highli inflect sentenc is one have lot of morphem for stem'
    ),
    'StopwordRemover' : (
        pp.StopwordRemover,
        'here is a lovely sentence that has copious stopwords',
        'lovely sentence copious stopwords'
    )
}
@pytest.mark.preprocess
@pytest.mark.parametrize(
    'complex_preprocessor',
    COMPLEX_PREPROCESSOR_PARAMETERS.values(),
    ids=COMPLEX_PREPROCESSOR_PARAMETERS.keys()
)
def test_complex_preprocessor(complex_preprocessor: Tuple[SinglePreprocessor, str, str]):
    """
    Test the preprocessor objects in `preprocess.preprocessors` that use more
    complex/varied/idiosyncratic mechanics.
    """

    complex_preprocessor_obj, raw_input, expected_output = complex_preprocessor
    raw_input = pd.DataFrame([raw_input], columns=['text'])

    computed_output = complex_preprocessor_obj().transform(raw_input)
    computed_output = computed_output.text.item()

    assert computed_output == expected_output


@pytest.mark.preprocess
def test_kitchen_sink_preprocessor(
    raw_data: dd.core.DataFrame,
    expected_preprocessed_data_kitchen_sink: dd.core.DataFrame
):
    """Test the `KitchenSinkPreprocessor` pipeline."""

    ksp = KitchenSinkPreprocessor()
    computed_preprocessed_data_kitchen_sink = ksp.preprocess(raw_data)

    dd.assert_eq(
        computed_preprocessed_data_kitchen_sink,
        expected_preprocessed_data_kitchen_sink
    )
