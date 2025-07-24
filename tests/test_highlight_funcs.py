import pytest
import tempfile
import os
from spurious_corr import utils


def test_highlight_dates_basic():
    text = "The dates are 2024-05-12 and 1999-12-31."
    matches = utils.highlight_dates(text)
    assert matches == ["2024-05-12", "1999-12-31"]


def test_highlight_dates_none():
    text = "No valid date here: 2024/05/12 or 05-12-2024."
    matches = utils.highlight_dates(text)
    assert matches == []


def test_highlight_from_list_basic():
    patterns = ["apple", "banana", "carrot"]
    highlight_func = utils.highlight_from_list(patterns)

    text = "I like apple and carrot juice."
    matches = highlight_func(text)

    assert sorted(matches) == ["apple", "carrot"]


def test_highlight_from_list_none():
    patterns = ["dog", "cat", "bird"]
    highlight_func = utils.highlight_from_list(patterns)

    text = "There is no animal here."
    matches = highlight_func(text)

    assert matches == []


def test_highlight_from_file(tmp_path):
    file_content = "red\ngreen\nblue\n"
    file_path = tmp_path / "colors.txt"

    with open(file_path, "w") as f:
        f.write(file_content)

    highlight_func = utils.highlight_from_file(file_path)

    text = "The flag has red and blue colors."
    matches = highlight_func(text)

    assert sorted(matches) == ["blue", "red"]


def test_highlight_html(tmp_path):
    # Simulate an html_tags.txt file:
    file_content = "<b> </b>\n<i> </i>\n<u> </u>\n"
    file_path = tmp_path / "html_tags.txt"

    with open(file_path, "w") as f:
        f.write(file_content)

    highlight_func = utils.highlight_html(file_path)

    text = "Here is some <b>bold</b> and <u>underlined</u> text."
    matches = highlight_func(text)

    # We expect it to find the raw tags present in file_content
    assert sorted(matches) == ["</b>", "</u>", "<b>", "<u>"]


def test_highlight_html_none(tmp_path):
    file_content = "<b> </b>\n<i> </i>\n<u> </u>\n"
    file_path = tmp_path / "html_tags.txt"

    with open(file_path, "w") as f:
        f.write(file_content)

    highlight_func = utils.highlight_html(file_path)

    text = "No html tags here"
    matches = highlight_func(text)

    assert matches == []
