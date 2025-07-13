"""A way to store and manipulate polynomials involving indeterminates."""

import re
from typing import Self

import numpy as np

from symboliccomputation.indeterminate import Indeterminate


class Monomial:
    """A monomial, like '-3*x^2*y^2'."""

    MONOMIAL_REGEX = re.compile(r"^-?\d*(\.\d+)?([a-z]+(\^[1-9]\d*)?)?(\*[a-z]+(\^[1-9]\d*)?)*$") # noqa: E501

    indeterminates: list[Indeterminate]
    coefficient: np.int64 | np.float64 # Only real numbers allowed right now

    def __init__(self, monomial: str) -> None:
        """Initialize an instance of the Monomial class.

        Args:
            monomial (str): The monomial desired, written like '-3*x^2*y^2'.
        """
        # Split the monomial string by "*", so each list element should be
        # something like either a number or an indeterminate raised to a power.

        # ValueError string to avoid duplication in traceback.
        value_error_msg = "Invalid monomial expression!"
        # Determine whether the given string is a valid monomial expression.
        if not Monomial.is_valid_monomial(monomial):
            raise ValueError(value_error_msg)

    @classmethod
    def is_valid_monomial(cls: Self, monomial: str) -> bool:
        """Test whether a monomial string is valid.

        It should contain potentially a number (potentially negative),
        followed potentially by an asterisk, a sequence of lowercase
        letters, and then potentially a "^" followed by a sequence of
        numbers. No fractional powers allowed.

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
