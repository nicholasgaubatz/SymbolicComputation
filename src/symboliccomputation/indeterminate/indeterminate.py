"""A way to store indeterminates, or the atoms of this package."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Indeterminate:
    """An indeterminate.

    Like $x$, to be used in a mathematical expression, where it doesn't
    represent a specific value.
    """

    name: str = "x"

    def __post_init__(self) -> None:
        """Ban indeterminate names from starting with numbers.

        This has the effect of banning monomials initialized like '12345x' (correct
        initialization string would be '12345*x').
        """
        if re.match(r"^\d", self.name):
            value_error_msg = f"Invalid name '{self.name}': must not start w/ digit."
            raise ValueError(value_error_msg)
