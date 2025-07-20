"""A way to store and manipulate polynomials involving indeterminates."""

import re
from typing import Self

import numpy as np

from symboliccomputation.indeterminate import Indeterminate


class Monomial:
    """A monomial, like '-3*x^2*y^2'."""

    # TODO(Nicholas): Monomial order comparisons given ordering, option to print
    # using given ordering,

    MONOMIAL_REGEX = re.compile(r"^(-\d+)?\d*(\.\d+)?([a-z]+(\^[1-9]\d*)?)?(\*[a-z]+(\^[1-9]\d*)?)*$") # noqa: E501

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
    
    
    # See https://stackoverflow.com/questions/2909106/whats-a-correct-and-good-way-to-implement-hash # noqa:
    def __key(self):
        """The unique key of a Monomial instance."""
        return (self.coefficient,
                tuple(sorted(self.weight_dict.items())))
    
    
    def __eq__(self, other: object) -> bool:
        """To test equality of two monomials."""
        if not isinstance(other, Monomial):
            return NotImplemented
        return self.__key() == other.__key()
    

    def __hash__(self) -> int:
        """To use this class in a set, need to be able to compute a hash."""
        return hash((self.coefficient, tuple(sorted(self.weight_dict.items()))))
    

    def __repr__(self) -> str:
        """Produce a string representation of the object.
        
        Details: If the Monomial has coefficient 1, it either just prints "1.0" or
        omits the coefficient, depending on whether the Monomial has any
        indeterminates. If an exponent is 1, it omits the exponent.
        """
        if np.float64(self.coefficient) == np.float64(1):
            if not self.weight_dict:
                return "1.0"
            monomial_string_list = []
        else:
            monomial_string_list = [str(self.coefficient)]

        for indeterminate, exponent in self.weight_dict.items():
            if np.float64(exponent) == np.float64(1):
                monomial_string_list.append(indeterminate.name)
            else:
                monomial_string_list.append(f"{indeterminate.name}^{exponent}")

        return "*".join(monomial_string_list)


class Polynomial:
    """A polynomial.

    Like f(x, y) = x^2 + 3xy - 2, writen as 'x^2 + 3*x*y - 2'.
    """

    indeterminates: set[Indeterminate]
    monomials: set[Monomial]

    def __init__(self, poly: str) -> None:
        """Initialize an instance of the Polynomial class.

        The syntax is very specific; maybe later I'll make it more general.

        Example:
        >> f = Polynomial("x^2 + 3*x*y - 2")

        Args:
            poly (str): The polynomial desired, written like 'f(x,y) = x^2 + 3*x*y - 2.'
        """
        # TODO(Nicholas): write docstring,
        # write operations like add, scalar multiply, multiply (big one), derivative,
        # integral, find roots if univariate. LONG-TERM: IDEAL, GROEBNER BASIS
        # A possible big goal is to make this also
        # able to handle matrices instead of determinates.
        # 001
        poly_split_by_spaces = poly.split(" ")
        self.monomials = set()

        # Test whether input string is valid.
        for i in range(len(poly_split_by_spaces)):
            token = poly_split_by_spaces[i]
            # Test whether all tokens are "+", "-", or a valid monomial.
            if token not in {"+", "-"} and not Monomial.is_valid_monomial_regex(token):
                value_error_msg = f"Invalid input: token '{token}'!"
                raise ValueError(value_error_msg)
            # Test that "+"/"-" and Monomial instances alternate.
            if i < len(poly_split_by_spaces)-1:
                token_1 = poly_split_by_spaces[i+1]
                if (token in {"+", "-"} and token_1 in {"+", "-"}) \
                or (Monomial.is_valid_monomial_regex(token) and \
                Monomial.is_valid_monomial_regex(token_1)):
                    value_error_msg = f"Invalid sequential tokens: {token}, {token_1}!"
                    raise ValueError(value_error_msg)

        # Create list of monomials.
        negate_next_monomial = False
        while len(poly_split_by_spaces) > 0:
            current_token = poly_split_by_spaces.pop(0)
            if current_token == "+":
                negate_next_monomial = False
                continue
            if current_token == "-":
                negate_next_monomial = True
                # poly_split_by_spaces[0] = "-" + poly_split_by_spaces[0]
            elif Monomial.is_valid_monomial_regex(current_token):
                current_monomial = Monomial(monomial=current_token)
                if negate_next_monomial:
                    current_monomial.coefficient *= -1.
                self.monomials.add(current_monomial)
                negate_next_monomial = False
            else:
                value_error_msg = "Issue with code: all cases should have been checked."
                raise ValueError(value_error_msg)

        # Create easily accessible list of the polynomial's indeterminates.
        self.indeterminates = set()
        for monomial in self.monomials:
            self.indeterminates.update(monomial.weight_dict.keys())
        self.indeterminates = set(self.indeterminates)
