"""Tests for the polynomial submodule."""

import pytest

from symboliccomputation.polynomial import Monomial


@pytest.mark.parametrize(argnames="valid_monomial", argvalues=[
    "0",
    "1",
    "2.0",
    "-3.0",
    "-4",
    "x",
    "asdf",
    "x^2",
    "x*y",
    "x*y^2",
    "x^2*y"
    "x^2*y^1234",
    "-5*x^3*y^321*z^8",
])
def test_valid_monomials(valid_monomial: list[str]) -> None:
    """Tests some valid monomials."""
    assert Monomial.is_valid_monomial(valid_monomial)


@pytest.mark.parametrize(argnames="invalid_monomial", argvalues=[
    "asdf1",
    "1.",
    "x^",
    "x2",
    "x^(1)",
    "x^-1",
    "x^0",
])
def test_invalid_monomials(invalid_monomial: list[str]) -> None:
    """Tests some invalid monomials."""
    assert not Monomial.is_valid_monomial(invalid_monomial)
