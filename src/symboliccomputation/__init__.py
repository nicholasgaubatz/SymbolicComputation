"""A simple symbolic computation package."""
from .indeterminate import Indeterminate
from .polynomial import Monomial, Polynomial
from .number import real

__all__ = ["Indeterminate", "Monomial", "Polynomial"]
