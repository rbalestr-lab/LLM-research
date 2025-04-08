"""
modifiers.py

This module defines the base Modifier class, as well as subclasses for injecting items 
(ItemInjection) and HTML tags (HTMLInjection) into text, as well as composing multiple 
modifiers (CompositeModifier).
"""

import random
import re

class Modifier:
    """
    Base class for applying modifications/corruptions to text-label pairs.

    Subclasses must implement the __call__ method to define specific transformations.
    
    Example:
        class MyModifier(Modifier):
            def __call__(self, text: str, label: Any) -> tuple[str, Any]:
                # custom transformation here
                return transformed_text, transformed_label
    """
    def __call__(self, text: str, label):
        """
        Apply the transformation to a single text-label pair.

        Args:
            text (str): The input text to transform.
            label: The associated label.
        
        Returns:
            tuple: (transformed_text, transformed_label)
        """
        raise NotImplementedError("Subclasses must implement __call__")

class CompositeModifier:
    """
    CompositeModifier chains multiple Modifier instances together.
    
    Each modifier from the list is applied sequentially to the text. This enables
    the combination of various transformations or injections into one composite operation.
    """
    def __init__(self, modifiers: list):
        """
        Initialize a CompositeModifier instance.
        
        Args:
            modifiers (list): A list of modifier instances (subclasses of Modifier) 
                              to be applied sequentially.
        """
        self.modifiers = modifiers

    def __call__(self, text: str, label):
        """
        Apply all modifiers in sequence to the given (text, label).

        Args:
            text (str): The input text.
            label: The associated label.

        Returns:
            tuple: The modified (text, label) pair after all transformations.
        """
        for modifier in self.modifiers:
            text, label = modifier(text, label)
        return text, label


class ItemInjection(Modifier):
    """
    A Modifier that injects items into text. 
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

    def __call__(self, text: str, label):
        """
        Inject tokens into the text at specified locations.

        Args:
            text (str): The input text to modify.
            label: The original label (unchanged).

        Returns:
            tuple: The modified text and the original label.
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
        return " ".join(words), label # return modified text and unchanged label

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
    

class HTMLInjection(Modifier):
    """
    A Modifier that injects HTML tags into a text.
    
    This injection picks a random tag from a file or list. Some items in the file/list
    have both opening and closing tags, while others only have a single tag.
    
    Injection behavior:
      - For "beginning": Inserts the opening tag at the beginning of the target text
        and, if available, inserts the closing tag at a random position.
      - For "random": Inserts the opening tag at a random position and, if available, inserts the
        closing tag at a random position after the opening tag.
      - For "end": Inserts the opening tag at a random position and, if available, appends the closing tag
        at the end of the target. If no closing tag is available, only the opening tag is inserted at the end.
      
    The optional parameter "level" allows the injection to be applied only inside the L-th nested
    HTML tag. For example:
        - If level=0, the injection is applied at the outermost level.
        - If level=2, and the text is "<a>asdf <b> asdf sadf asdf </b> asdf asdf </a>", then the injection
          is performed only within the inner <b>...</b> region.
    """
    def __init__(self, file_path: str, location: str = "random", level: int = None):
        """
        Initialize an HTMLInjection instance.
        
        Args:
            file_path (str): Path to the file containing HTML tag definitions.
            location (str): Where to inject ("beginning", "random", or "end").
            level (int): The nested HTML level at which to perform the injection. 
                        None means anywhere in the text.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.location = location
        self.level = level
    
    @classmethod
    def from_file(cls, file_path: str, location: str = "random", level: int = None):
        """
        Create an HTMLInjection instance by reading tag definitions from a file.
        
        Args:
            file_path (str): Path to the file containing HTML tag definitions.
            location (str): Where to inject ("beginning", "random", or "end").
            level (int): The nested HTML level at which to perform the injection.
        
        Returns:
            HTMLInjection: Configured instance using tags from the file.
        """
        return cls(file_path, location=location, level=level)

    @classmethod
    def from_list(cls, tags: list, location: str = "random", level: int = None):
        """
        Create an HTMLInjection instance using a predefined list of tag definitions.
        
        Args:
            tags (list): A list of strings, each defining an HTML tag injection. Each string should
                         either contain one token (a single tag) or two tokens (opening and closing tags).
            location (str): Where to inject ("beginning", "random", or "end").
            level (int): The nested HTML level at which to perform the injection.
        
        Returns:
            HTMLInjection: Configured instance using the provided list of tags.
        """
        instance = cls.__new__(cls)
        instance.tags = tags
        instance.location = location
        instance.level = level
        return instance

    def _choose_tag(self):
        """
        Randomly choose a tag from the loaded list.
        
        Returns:
            tuple: (opening_tag, closing_tag) if available; if only one token is provided,
                   returns (tag, None).
        """
        line = random.choice(self.tags)
        parts = line.split()
        if len(parts) >= 2:
            return parts[0], parts[1]
        else:
            return parts[0], None

    def _inject_into_tokens(self, tokens, opening, closing, location):
        """
        Injects the opening (and optionally closing) tag into a list of tokens based on the location.
        
        Args:
            tokens (list): List of tokens (words) from the target substring.
            opening (str): The opening HTML tag to inject.
            closing (str or None): The closing HTML tag to inject, if available.
            location (str): Injection location ("beginning", "random", or "end").
        
        Returns:
            list: New list of tokens with the tags injected.
        """
        if location == "beginning":
            new_tokens = [opening] + tokens[:]
            if closing:
                pos = random.randint(1, len(new_tokens))
                new_tokens.insert(pos, closing)
            return new_tokens
        elif location == "end":
            new_tokens = tokens[:]
            if closing:
                pos = random.randint(0, len(new_tokens))
                new_tokens.insert(pos, opening)
                new_tokens.append(closing)
            else:
                new_tokens.append(opening)
            return new_tokens
        elif location == "random":
            new_tokens = tokens[:]
            pos_open = random.randint(0, len(new_tokens))
            new_tokens.insert(pos_open, opening)
            if closing:
                # Ensure closing tag is inserted after opening tag.
                pos_close = random.randint(pos_open + 1, len(new_tokens))
                new_tokens.insert(pos_close, closing)
            return new_tokens
        else:
            return tokens

    def _inject(self, text, location):
        """
        Performs injection on the given text (operating on the entire text).
        
        Args:
            text (str): The text to modify.
            location (str): The injection location.
        
        Returns:
            str: The text with injected HTML tag(s).
        """
        tokens = text.split()
        opening, closing = self._choose_tag()
        new_tokens = self._inject_into_tokens(tokens, opening, closing, location)
        return " ".join(new_tokens)

    def _find_level_span(self, text, level):
        """
        Finds the first content span corresponding to the specified nesting level.
        Assumes well-formed HTML.
        
        Args:
            text (str): The HTML text.
            level (int): The desired nesting level (0 for outermost, 1 for first tag, etc.).
        
        Returns:
            tuple or None: (start_index, end_index) of the content inside the tag at the given level,
                           or None if not found.
        """
        tag_regex = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)[^>]*>")
        stack = []
        for match in tag_regex.finditer(text):
            tag_str = match.group(0)
            tag_name = match.group(1)
            if not tag_str.startswith("</"):
                # Opening tag: push (tag_name, end_index of opening tag)
                stack.append((tag_name, match.end()))
            else:
                if stack:
                    open_tag, start_index = stack.pop()
                    # When the tag is closed, if the length of the stack becomes target_level - 1,
                    # then the closed tag was at the target level.
                    if len(stack) == level - 1:
                        span_start = start_index
                        span_end = match.start()
                        return (span_start, span_end)
        return None
    
    def __call__(self, text: str, label):
        """
        Injects HTML tag(s) into the text.
        
        If self.level is None, injection is performed on the entire text. Otherwise, the injection
        is applied only within the first occurrence of a nested tag at the specified level.
        
        Args:
            text (str): The input text to modify.
            label: The original label (unchanged).

        Returns:
            tuple: The modified text and the original label.
        """
        if self.level is None:
            return self._inject(text, self.location), label
        elif self.level == 0:
            # Wrap the entire text with the injection.
            opening, closing = self._choose_tag()
            if closing:
                return f"{opening}{text}{closing}", label
            else:
                return f"{opening}{text}{opening}", label
        else:
            span = self._find_level_span(text, self.level)
            if span is None:
                # Fallback to free injection if the specified level is not found.
                return self._inject(text, self.location), label
            start, end = span
            target = text[start:end]
            injected = self._inject(target, self.location)
            return text[:start] + injected + text[end:], label
