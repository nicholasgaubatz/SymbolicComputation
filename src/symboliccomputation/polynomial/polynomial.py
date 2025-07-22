"""A way to store and manipulate polynomials involving indeterminates."""

import re
from typing import Self

import numpy as np

from symboliccomputation.indeterminate import Indeterminate


class Monomial:
    """A monomial, like '-3*x^2*y^2'."""

    # TODO(Nicholas): Monomial order comparisons given ordering, option to print
    # using given ordering,
    # 003

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

    # See https://stackoverflow.com/questions/2909106/whats-a-correct-and-good-way-to-implement-hash # noqa: E501
    def __key(self) -> tuple[np.float64, tuple[np.int64]]:
        """Get a unique key of a Monomial instance."""
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
            monomial_string_list = [str(np.float64(self.coefficient))]

        for indeterminate, exponent in self.weight_dict.items():
            if np.float64(exponent) == np.float64(1):
                monomial_string_list.append(indeterminate.name)
            else:
                monomial_string_list.append(f"{indeterminate.name}^{exponent}")

        return "*".join(monomial_string_list)

    def __mul__(self, other: object) -> Self:
        """Multiply two Monomials by essentially adding their weight_dicts."""
        if not isinstance(other, Monomial):
            return NotImplemented

        new_monomial = Monomial("0")
        new_monomial.coefficient = self.coefficient * other.coefficient
        for key in self.weight_dict:
            new_monomial.weight_dict[key] = self.weight_dict[key]

        for key in other.weight_dict:
            if key in new_monomial.weight_dict:
                new_monomial.weight_dict[key] += other.weight_dict[key]
            else:
                new_monomial.weight_dict[key] = other.weight_dict[key]

        if set(new_monomial.weight_dict.values()) == {0} \
        or new_monomial.coefficient == 0:
            return Monomial(str(new_monomial.coefficient))

        return new_monomial

    def derivative(self, wrt: str | Indeterminate) -> Self:
        """Take the derivative of the monomial with respect to an indeterminate.

        Args:
            wrt (str | Indeterminate): The indeterminate to take the derivative
                w.r.t., either the name or the Indeterminate itself.

        Returns:
            Self: A new Monomial consisting of the derivative.
        """
        if isinstance(wrt, str):
            wrt = Indeterminate(wrt)

        # If wrt doesn't show up in the monomial's indeterminate list, simply return 0.
        if wrt not in set(self.weight_dict.keys()):
            return Monomial("0")

        exponent = self.weight_dict[wrt]
        new_monomial = Monomial("0")
        new_monomial.coefficient = self.coefficient * exponent
        new_monomial.weight_dict = self.weight_dict
        new_monomial.weight_dict[wrt] -= 1
        if new_monomial.weight_dict[wrt] == 0:
            del new_monomial.weight_dict[wrt]

        return new_monomial


class Polynomial:
    """A polynomial.

    Like f(x, y) = x^2 + 3xy - 2, writen as 'x^2 + 3*x*y - 2'.
    """

    # TODO(Nicholas): write docstring,
        # write operations like integral (need to implement fractions and irrationals!)
        # find roots if univariate. LONG-TERM: IDEAL, GROEBNER BASIS
        # A possible big goal is to make this also
        # able to handle matrices instead of determinates.
        # 001

    indeterminates: set[Indeterminate]
    monomials: set[Monomial]

    def __init__(self, poly: str | set[Monomial]) -> None:
        """Initialize an instance of the Polynomial class.

        The syntax is very specific; maybe later I'll make it more general.

        Example:
        >> f = Polynomial("x^2 + 3*x*y - 2")

        Args:
            poly (str): The polynomial desired, written like 'f(x,y) = x^2 + 3*x*y - 2.'
        """
        if isinstance(poly, str):
            poly_split_by_spaces = poly.split(" ")
            self.monomials = set()

            # Test whether input string is valid.
            for i in range(len(poly_split_by_spaces)):
                token = poly_split_by_spaces[i]
                # Test whether all tokens are "+", "-", or a valid monomial.
                if token not in {"+", "-"} \
                and not Monomial.is_valid_monomial_regex(token):
                    value_error_msg = f"Invalid input: token '{token}'!"
                    raise ValueError(value_error_msg)
                # Test that "+"/"-" and Monomial instances alternate.
                if i < len(poly_split_by_spaces)-1:
                    token_1 = poly_split_by_spaces[i+1]
                    if (token in {"+", "-"} and token_1 in {"+", "-"}) \
                    or (Monomial.is_valid_monomial_regex(token) and \
                    Monomial.is_valid_monomial_regex(token_1)):
                        value_error_msg = f"Invalid sequential tokens: {token}, {token_1}!" # noqa: E501
                        raise ValueError(value_error_msg)

            # Create list of monomials.
            monomial_list = []
            negate_next_monomial = False
            while len(poly_split_by_spaces) > 0:
                current_token = poly_split_by_spaces.pop(0)
                if current_token == "+":
                    negate_next_monomial = False
                    continue
                if current_token == "-":
                    negate_next_monomial = True
                elif Monomial.is_valid_monomial_regex(current_token):
                    current_monomial = Monomial(monomial=current_token)
                    if negate_next_monomial:
                        current_monomial.coefficient *= -1.
                    monomial_list.append(current_monomial)
                    negate_next_monomial = False
                else:
                    value_error_msg = "Code issue: all cases should have been checked."
                    raise ValueError(value_error_msg)

            # Check that no monomials repeat, even up to scalar.
            # We choose to prohibit this.
            # TODO(Nicholas): See whether we should allow this and
            # just use consolidation method below.
            # 002
            for i in range(len(monomial_list)):
                for j in range(i+1, len(monomial_list)):
                    m1, m2 = monomial_list[i], monomial_list[j]
                    if m1.weight_dict == m2.weight_dict:
                        value_error_msg = f"Invalid input: repeated monomials {m1}, {m2}!" # noqa: E501
                        raise ValueError(value_error_msg)

            # Now that we know there are no repeats, convert monomial list to set.
            self.monomials = set(monomial_list)

        # Should only be used internally.
        elif isinstance(poly, set) and all(isinstance(m, Monomial) for m in poly):
            self.monomials = poly

        # In both cases, remove all monomials with coefficient 0.
        self.monomials = {m for m in list(self.monomials) if m.coefficient != 0}

        # If polynomial is empty, make it the zero polynomial.
        if not self.monomials:
            self.monomials = {Monomial("0")}

        # Create easily accessible list of the polynomial's indeterminates.
        self.indeterminates = set()
        for monomial in self.monomials:
            self.indeterminates.update(monomial.weight_dict.keys())

    def __eq__(self, other: object) -> bool:
        """Test equality of two polynomials.

        Two polynomials are equal iff their monomial sets are equal.
        """
        if not isinstance(other, Polynomial):
            return NotImplemented

        return self.monomials == other.monomials

    def __hash__(self) -> int:
        """To use this class in a set, need to be able to compute a hash."""
        return hash((self.indeterminates, self.monomials))

    def __repr__(self) -> str:
        """Produce a string representation of the object.

        Details: Simply prints the sum of all monomials of the polynomial.
        """
        return " + ".join([m.__repr__() for m in self.monomials])

    def __add__(self, other: object) -> Self:
        """Add two polynomials, regardless of their indeterminates."""
        if not isinstance(other, Polynomial):
            return NotImplemented

        # Get the polynomials' sets of monomials as lists and concatenate.
        all_monomials = list(self.monomials) + list(other.monomials)

        # For each monomial pair, if underlying weight_dicts are equal, add their
        # coefs and remove the second monomial from consideration.
        final_monomial_set = Polynomial._consolidate_monomial_list(all_monomials)

        return Polynomial(final_monomial_set)

    def __sub__(self, other: object) -> Self:
        """Subtract two polynomials."""
        if not isinstance(other, Polynomial):
            return NotImplemented

        # Negate the other polynomial.
        for m in other.monomials:
            m.coefficient *= -1

        return self + other

    def __mul__(self, other: object) -> Self:
        """Multiply two polynomials."""
        if not isinstance(other, Polynomial):
            return NotImplemented

        new_monomial_list = [m1*m2 for m1 in self.monomials for m2 in other.monomials]

        final_monomial_set = Polynomial._consolidate_monomial_list(new_monomial_list)

        return Polynomial(final_monomial_set)

    def __rmul__(self, other: object) -> Self:
        """Handle left scalar multiplication."""
        if type(other) not in {int, float, np.int64, np.float64}:
            return NotImplemented

        return Polynomial(str(other)) * self

    def derivative(self, wrt: str | Indeterminate) -> Self:
        """Take the derivative of the polynomial with respect to an indeterminate.

        Args:
            wrt (str | Indeterminate): The indeterminate to take the derivative w.r.t.

        Returns:
            Self: A new Polynomial consisting of the derivative.
        """
        if isinstance(wrt, str):
            wrt = Indeterminate(wrt)

        # If wrt doesn't show up in the polynomial's indeterminate list, simply return 0
        if wrt not in self.indeterminates:
            return Polynomial("0")

        new_monomial_set = set()

        for m in self.monomials:
            m_prime = m.derivative(wrt=wrt)
            if m_prime != Monomial("0"):
                new_monomial_set.add(m_prime)

        return Polynomial(new_monomial_set)

    @classmethod
    def _consolidate_monomial_list(cls, monomial_list: list[Monomial]) -> set[Monomial]:
        """Add monomials with equal weight vectors. Class method.

        Args:
            monomial_list (list[Monomial]): List of monomials, possibly with repeat
                                            weight vectors.

        Returns:
            set[Monomial]: List of monomials with no repeat weight vectors.
        """
        final_monomial_set = set()
        monomial_covered_mask = [False]*len(monomial_list)

        for i in range(len(monomial_list)):
            if not monomial_covered_mask[i]:
                m1 = monomial_list[i]
                for j in range(i+1, len(monomial_list)):
                    m2 = monomial_list[j]
                    if m1.weight_dict == m2.weight_dict:
                        monomial_covered_mask[j] = True
                        monomial_list[i].coefficient += m2.coefficient
                final_monomial_set.add(m1)

        return final_monomial_set
