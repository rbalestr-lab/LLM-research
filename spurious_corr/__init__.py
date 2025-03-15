"""
spurious_corr package

This package provides tools to apply and test the effect of various transformations 
(such as injecting spurious text) on datasets for research and testing purposes. 
It includes functionality for text transformations, various generators for spurious 
text, and utilities for printing and highlighting text.
"""

from .modifiers import Modifier, CompositeModifier, ItemInjection, HTMLInjection
from .transform import spurious_transform
from .generators import (
    spurious_date_generator
)
from .utils import pretty_print, pretty_print_dataset, highlight_dates, highlight_from_file, highlight_html
