"""A way to handle real numbers, both rational and complex."""

import math

import numpy as np

class Rational:
    """A rational number a/b.
    """

    numerator: np.int64
    denominator: np.int64

    def __init__(self,
                 numerator: np.int64 | None = None,
                 denominator: np.int64 | None = None,
                 decimal: np.float64 | None = None) -> None:
        """Constructs a rational number in simplified form.

        At least one of (numerator/denominator) and decimal should be None.
        If the number is negative, the negative part will always be stored in the
        numerator.

        Args:
            numerator (np.int64 | None): The numerator.
            denominator (np.int64 | None): The denominator.
            decimal (np.float64 | None): The decimal representation of the number.
        Raises:
            ValueError: If incorrect combinations of arguments are given.
            ZeroDivisionError: If denominator is 0 (or close to it).
        """
        if decimal is None:
            if numerator is None or denominator is None:
                value_error_msg = "Error: if decimal is None, num/denom must not be."
                raise ValueError(value_error_msg)
            if np.isclose(denominator, np.int64(0)):
                raise ZeroDivisionError

            numerator, denominator = np.int64(numerator), np.int64(denominator)
            gcd = math.gcd(numerator, denominator)

            if denominator < 0.:
                numerator *= -1.
                denominator *= -1.

            self.numerator = np.int64(numerator / gcd)
            self.denominator = np.int64(denominator / gcd)
        else:
            if numerator is not None or denominator is not None:
                value_error_msg = "if decimal is not None, num/denom must be."
                raise ValueError(value_error_msg)
            
            decimal = np.float64(decimal)
            decimal_str = str(decimal)
            sig_figs = decimal_str[::-1].find(".")
            ten_appropriate_power = 10**sig_figs

            self.numerator = np.int64(decimal*ten_appropriate_power)
            self.denominator = np.int64(ten_appropriate_power)
            

    def simplify(self) -> None:
        """Simplify the rational number in-place."""
        gcd = math.gcd(self.numerator, self.denominator)
        self.numerator /= gcd
        self.denominator /= gcd