"""Tests for the real submodule."""

import math
import numpy as np
import pytest

from symboliccomputation.number.real import Rational

########################################################################################
# Rationals
########################################################################################

@pytest.mark.parametrize(argnames=("numerator", "denominator", "exp_num", "exp_denom"),
                         argvalues=[
    (0, 1, 0, 1),
    (0, -1, 0, 1),
    (1, 1, 1, 1),
    (2, 2, 1, 1),
    (9, 60, 3, 20),
    (-9, 60, -3, 20),
    (9, -60, -3, 20),
    (-9, -60, 3, 20),
])
def test_valid_rational_fraction(numerator: list[np.int64],
                                 denominator: list[np.int64],
                                 exp_num: list[np.int64],
                                 exp_denom: list[np.int64]) -> None:
    """Tests whether fraction method of rational construction works."""
    rat = Rational(numerator=numerator, denominator=denominator)
    assert rat.numerator == exp_num and rat.denominator == exp_denom


@pytest.mark.parametrize(argnames=("decimal", "exp_num", "exp_denom"),
                         argvalues=[
    (0.0, 0, 10),
    (0.00, 0, 10),
    (0.1, 1, 10),
    (3.14, 314, 100),
    (-3.14, -314, 100),
    (12345.12345, 1234512345, 100000),
])
def test_valid_rational_decimal(decimal: list[np.float64],
                                exp_num: list[np.int64],
                                exp_denom: list[np.int64]) -> None:
    """Tests whether decimal method of rational construction works."""
    rat = Rational(decimal=decimal)
    assert rat.numerator == exp_num and rat.denominator == exp_denom
