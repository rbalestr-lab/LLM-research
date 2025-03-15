"""
modifiers.py

This module defines the base Modifier class and the ItemInjection class for
injecting spurious tokens into text. It supports multiple injection sources:
from a list, a file, or a function that generates a new injection item each time.
"""

import random

class Modifier:
    """
    Base class for applying modifications/corruptions to text.

    Subclasses must implement the __call__ method to define specific transformations.
    
    Example:
        class MyModifier(Modifier):
            def __call__(self, text: str) -> str:
                # custom transformation here
                return transformed_text
    """
    def __call__(self, text: str) -> str:
        """
        Apply the transformation to a single text instance.

        Args:
            text (str): The input text to transform.
        
        Returns:
            str: The transformed text.
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    
class ItemInjection(Modifier):
    """
    A Modifier that injects items into text. The injection source is a callable
    that returns a new injection item each time it is called.

    This class supports creation via three different approaches:
    - from_list: Using a predefined list of injection items.
    - from_file: Reading injection items from a file.
    - from_function: Using a custom function to generate injections.
    """
    def __init__(self, injection_source, location: str = "random", token_proportion: float = 0.1):
        """
        Initialize an ItemInjection instance.

        Args:
            injection_source (callable): A function that returns an injection token.
            location (str): Where to inject the token ("beginning", "random", "end").
            token_proportion (float): Proportion of tokens in the text to be affected.
        """
        assert callable(injection_source), "injection_source must be callable"
        self.injection_source = injection_source
        self.location = location
        self.token_proportion = token_proportion

        assert 0 <= token_proportion <= 1, "token_proportion must be between 0 and 1"
        assert location in {"beginning", "random", "end"}, "location must be 'beginning', 'random', or 'end'"

    def __call__(self, text: str) -> str:
        """
        Inject spurious tokens into the text.

        Args:
            text (str): The original text.
        
        Returns:
            str: The text with injected tokens.
        """
        words = text.split()
        num_tokens = len(words)
        # Ensure at least one token is injected
        num_to_inject = max(1, int(num_tokens * self.token_proportion))
        injections = [self.injection_source() for _ in range(num_to_inject)]
        
        if self.location == "beginning":
            words = injections + words
        elif self.location == "end":
            words = words + injections
        elif self.location == "random":
            for injection in injections:
                pos = random.randint(0, len(words))
                words.insert(pos, injection)
        return " ".join(words)

    @classmethod
    def from_list(cls, items: list, location: str = "random", token_proportion: float = 0.1):
        """
        Create an ItemInjection instance using a predefined list of injection items.

        Args:
            items (list): List of tokens (strings) to choose from.
            location (str): Where to inject the tokens ("beginning", "random", "end").
            token_proportion (float): Proportion of the text tokens to be affected.
        
        Returns:
            ItemInjection: Configured instance with a random choice injection source.
        """
        def injection_source():
            return random.choice(items)
        return cls(injection_source, location=location, token_proportion=token_proportion)

    @classmethod
    def from_file(cls, file_path: str, location: str = "random", token_proportion: float = 0.1):
        """
        Create an ItemInjection instance using items read from a file.
        Each non-empty line in the file is treated as an injection item.

        Args:
            file_path (str): Path to the file containing injection items.
            location (str): Where to inject the tokens ("beginning", "random", "end").
            token_proportion (float): Proportion of the text tokens to be affected.
        
        Returns:
            ItemInjection: Configured instance using tokens from the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            items = [line.strip() for line in file if line.strip()]
        return cls.from_list(items, location=location, token_proportion=token_proportion)

    @classmethod
    def from_function(cls, injection_func, location: str = "random", token_proportion: float = 0.1):
        """
        Create an ItemInjection instance using a custom function that generates
        a new injection item each time it is called.

        Args:
            injection_func (callable): Function that returns an injection item.
            location (str): Where to inject the token ("beginning", "random", "end").
            token_proportion (float): Proportion of the text tokens to be affected.
        
        Returns:
            ItemInjection: Configured instance using the provided function.
        """
        assert callable(injection_func), "injection_func must be callable"
        return cls(injection_func, location=location, token_proportion=token_proportion)