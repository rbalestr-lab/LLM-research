"""
generators.py

This module provides generator functions for creating spurious text injections.
These functions can be used directly or integrated with the ItemInjection modifier.
"""

import random

def spurious_date_generator():
    """
    Generates a random date in the format YYYY-MM-DD to use as a spurious injection.

    Returns:
        str: A randomly generated date string.
    """
    year = random.randint(1900, 2100)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # To avoid invalid dates
    return f"{year}-{month:02d}-{day:02d}"

