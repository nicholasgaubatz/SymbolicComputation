"""A way to store and manipulate polynomials involving indeterminates."""

import re
from typing import Self

import numpy as np

from symboliccomputation.indeterminate import Indeterminate


class Monomial:
    """A monomial, like '-3*x^2*y^2'."""

    MONOMIAL_REGEX = re.compile(r"^-?\d*(\.\d+)?([a-z]+(\^[1-9]\d*)?)?(\*[a-z]+(\^[1-9]\d*)?)*$") # noqa: E501

    coefficient: np.int64 | np.float64 # Only real numbers allowed right now
    weight_dict: dict[Indeterminate, np.int64]

    def __init__(self, monomial: str) -> None:
        """Initialize an instance of the Monomial class.

        Currently, only handles indeterminates in alphabetical order. Later, want to
        implement a given monomial ordering.

        Args:
            monomial (str): The monomial desired, written like '-3*x^2*y^2'.
        """
        # Determine whether the given string is a valid monomial expression.
        if not Monomial.is_valid_monomial_regex(monomial):
            # ValueError string to avoid duplication in traceback.
            value_error_msg = "Invalid monomial expression!"
            raise ValueError(value_error_msg)

        # Split the monomial string by "*", so each list element should be
        # something like either a number or an indeterminate raised to a power.
        monomial_split_by_asterisk = monomial.split("*")

        # Remove and store the monomial's coefficient, if it has one.
        try:
            np.float64(monomial_split_by_asterisk[0])
            self.coefficient = np.float64(monomial_split_by_asterisk.pop(0))
        except ValueError:
            self.coefficient = np.float64(1.)

        # Sort the list according to indeterminate name.
        monomial_split_by_asterisk.sort()

        # Split each list element at "^", so each element of the list is
        # [indeterminate, exponent].
        monomial_fully_split = [elt.split("^") for elt in monomial_split_by_asterisk]

        # If any indeterminates have no exponent, make them have exponent 1.
        monomial_fully_split_ones_added = [
            [*elt, "1"] if len(elt) == 1 else elt for elt in monomial_fully_split
        ]

        # Determine all the monomial's indeterminates and raise error if any repeats.
        indeterminate_string_list = [elt[0] for elt in monomial_fully_split_ones_added]
        if len(list(set(indeterminate_string_list))) != len(indeterminate_string_list):
            value_error_msg = "Invalid monomial expression: no repeat indeterminates!"
            raise ValueError(value_error_msg)

        # Determine the monomial's weight vector.
        self.weight_dict = {Indeterminate(elt[0]): np.int64(elt[1])
                              for elt in monomial_fully_split_ones_added}


    @classmethod
    def is_valid_monomial_regex(cls: Self, monomial: str) -> bool:
        """Test whether a monomial string is valid.

        It should contain potentially a number (potentially negative),
        followed potentially by an asterisk, a sequence of lowercase
        letters, and then potentially a "^" followed by a sequence of
        numbers. No fractional powers allowed. No repeat indeterminates allowed.

        Args:
            monomial (str): The string to compare.

        Returns:
            bool: Whether the monomial string is valid.
        """
        return bool(cls.MONOMIAL_REGEX.match(monomial))


class Polynomial:
    """A polynomial.

    Like f(x, y) = x^2 + 3xy - 2, writen as 'f(x,y) = x^2 + 3*x*y - 2'.
    """

    indeterminates: list[Indeterminate]
    monomials: list[Monomial]

    def __init__(self, poly: str) -> None:
        """Initialize an instance of the Polynomial class.

        The syntax is very specific; maybe later I'll make it more general.

        Example:
        >> f = Polynomial("x^2 + 3*x*y - 2")

        Args:
            poly (str): The polynomial desired, written like 'f(x,y) = x^2 + 3*x*y - 2.'
        """
        # TODO(Nicholas): write docstring, check that all indeterminates are actually
        # indeterminates, write attributes like indeterminates contained,
        # write operations like add, scalar multiply, multiply (big one), derivative,
        # integral, find roots if univariate. A bigger goal is to make this also
        # able to handle matrices instead of determinates.
        # 001
