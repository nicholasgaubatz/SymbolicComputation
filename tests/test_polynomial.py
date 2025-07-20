"""Tests for the polynomial submodule."""

import re

import numpy as np
import pytest

from symboliccomputation.indeterminate import Indeterminate
from symboliccomputation.polynomial import Monomial, Polynomial


####################################
# Monomials
####################################


@pytest.mark.parametrize(argnames="valid_monomial_regex", argvalues=[
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
def test_valid_monomial_regex(valid_monomial_regex: list[str]) -> None:
    """Tests some valid monomial regular expressions."""
    assert Monomial.is_valid_monomial_regex(valid_monomial_regex)


@pytest.mark.parametrize(argnames="invalid_monomial_regex", argvalues=[
    "asdf1",
    "1.",
    "x^",
    "x2",
    "x^(1)",
    "x^-1",
    "x^0",
])
def test_invalid_monomial_regex(invalid_monomial_regex: list[str]) -> None:
    """Tests some invalid monomial regular expressions."""
    assert not Monomial.is_valid_monomial_regex(invalid_monomial_regex)


@pytest.mark.parametrize(argnames="invalid_monomial", argvalues=[
    "asdf1",
    "1.",
    "x^",
    "x2",
    "x^(1)",
    "x^-1",
    "x^0",
    "-"
])
def test_invalid_monomials(invalid_monomial: list[str]) -> None:
    """Tests some invalid monomial regular expressions in the Monomial class itself."""
    with pytest.raises(ValueError, match="Invalid monomial expression!") as exc_info:
        Monomial(invalid_monomial)
    assert "Invalid monomial expression!" in str(exc_info.value)


@pytest.mark.parametrize(argnames="invalid_monomial_repeated_ind", argvalues=[
    "x*x",
    "1.0*xy*xy^2",
    "as^3*as^2",
    "wow^3*x*wow",
])
def test_invalid_monomials_rep_ind(invalid_monomial_repeated_ind: list[str]) -> None:
    """Tests some invalid monomial regular expressions in the Monomial class itself."""
    with pytest.raises(ValueError, match=("Invalid monomial expression: "
                       "no repeat indeterminates!")) as exc_info:
        Monomial(invalid_monomial_repeated_ind)
    assert "Invalid monomial expression: no repeat indeterminates!" in str(
        exc_info.value)


@pytest.mark.parametrize(argnames=("valid_monomial", "exp_coef", "exp_weights"),
                         argvalues=[
    ("0", np.float64(0), {}),
    ("1", np.float64(1), {}),
    ("2.0", np.float64(2), {}),
    ("-3.0", np.float64(-3), {}),
    ("-4", np.float64(-4), {}),
    ("x", np.float64(1), {Indeterminate("x"): np.int64(1)}),
    ("asdf", np.float64(1), {Indeterminate("asdf"): np.int64(1)}),
    ("x^2", np.float64(1), {Indeterminate("x"): np.int64(2)}),
    ("x*y", np.float64(1), {Indeterminate("x"): np.int64(1),
                            Indeterminate("y"): np.int64(1)}),
    ("x*y^2", np.float64(1), {Indeterminate("x"): np.int64(1),
                              Indeterminate("y"): np.int64(2)}),
    ("x^2*y", np.float64(1), {Indeterminate("x"): np.int64(2),
                              Indeterminate("y"): np.int64(1)}),
    ("x^2*y^1234", np.float64(1), {Indeterminate("x"): np.int64(2),
                                   Indeterminate("y"): np.int64(1234)}),
    ("-5*x^3*y^321*z^8", np.float64(-5), {Indeterminate("x"): np.int64(3),
                                          Indeterminate("y"): np.int64(321),
                                          Indeterminate("z"): np.int64(8)}),
    ("-5*y^321*z^8*x^3", np.float64(-5), {Indeterminate("x"): np.int64(3),
                                          Indeterminate("y"): np.int64(321),
                                          Indeterminate("z"): np.int64(8)}),
])
def test_valid_monomial(valid_monomial: list[str],
                              exp_coef: np.float64,
                              exp_weights: dict[Indeterminate, np.int64]) -> None:
    """Tests some valid monomial regular expressions."""
    current_monomial = Monomial(valid_monomial)

    # Assert coefficient is correct.
    assert current_monomial.coefficient == exp_coef

    # Assert weight_dict is correct.
    assert current_monomial.weight_dict == exp_weights


@pytest.mark.parametrize(argnames=("monomial", "monomial_str_repr"),
                         argvalues=[
    ("0", "0.0"),
    ("1", "1.0"),
    ("2.0", "2.0"),
    ("-3.0", "-3.0"),
    ("-4", "-4.0"),
    ("x", "x"),
    ("asdf", "asdf"),
    ("x^2", "x^2"),
    ("x*y", "x*y"),
    ("x*y^2", "x*y^2"),
    ("x^2*y", "x^2*y"),
    ("x^2*y^1234", "x^2*y^1234"),
    ("-5*x^3*y^321*z^8", "-5.0*x^3*y^321*z^8"),
    ("-5*y^321*z^8*x^3", "-5.0*x^3*y^321*z^8"),
])
def test_monomial_repr(monomial: list[str],
                       monomial_str_repr: list[str]) -> None:
    """Tests some Monomial string representations."""
    current_monomial = Monomial(monomial)
    assert current_monomial.__repr__() == monomial_str_repr


####################################
# Polynomials
####################################


@pytest.mark.parametrize(argnames=("invalid_polynomial", "exp_error_msg"), argvalues=[
    ("asdf1", "Invalid input: token 'asdf1'!"),
    ("1.", "Invalid input: token '1.'!"),
    ("x^", "Invalid input: token 'x^'!"),
    ("x2", "Invalid input: token 'x2'!"),
    ("x^(1)", "Invalid input: token 'x^(1)'!"),
    ("x^-1", "Invalid input: token 'x^-1'!"),
    ("x^0", "Invalid input: token 'x^0'!"),
    ("x ++ y", "Invalid input: token '++'!"),
    ("x y", "Invalid sequential tokens: x, y!")
    # ("", "Invalid input: token ''!"),
])
def test_invalid_polynomials(invalid_polynomial: list[str],
                             exp_error_msg: list[str]) -> None:
    """Tests some invalid polynomial init strings in the Polynomial class itself."""
    with pytest.raises(ValueError, match=re.escape(exp_error_msg)) as exc_info:
        Polynomial(invalid_polynomial)
    assert exp_error_msg in str(exc_info.value)


@pytest.mark.parametrize(argnames=("valid_polynomial",
                                   "exp_indeterminates",
                                   "exp_monomials"),
                         argvalues=[
    ("0", set(), {Monomial("0")}),
    ("1", set(), {Monomial("1")}),
    ("2.0", set(), {Monomial("2.0")}),
    ("-3.0", set(), {Monomial("-3.0")}),
    ("-4", set(), {Monomial("-4")}),
    ("x", {Indeterminate("x")}, {Monomial("x")}),
    ("asdf", {Indeterminate("asdf")}, {Monomial("asdf")}),
    ("x^2", {Indeterminate("x")}, {Monomial("x^2")}),
    ("x*y", {Indeterminate("x"), Indeterminate("y")}, {Monomial("x*y")}),
    ("x*y^2", {Indeterminate("x"), Indeterminate("y")}, {Monomial("x*y^2")}),
    ("x^2*y", {Indeterminate("x"), Indeterminate("y")}, {Monomial("x^2*y")}),
    ("x^2*y^1234",
     {Indeterminate("x"), Indeterminate("y")},
     {Monomial("x^2*y^1234")}),
    ("-5*x^3*y^321*z^8",
     {Indeterminate("x"), Indeterminate("y"), Indeterminate("z")},
     {Monomial("-5*x^3*y^321*z^8")}),
    ("-5*y^321*z^8*x^3",
     {Indeterminate("x"), Indeterminate("y"), Indeterminate("z")},
     {Monomial("-5*y^321*z^8*x^3")}),
])
def test_valid_monomial(valid_polynomial: list[str],
                        exp_indeterminates: set[Indeterminate],
                        exp_monomials: set[Monomial]) -> None:
    """Tests some valid monomial regular expressions."""
    current_polynomial = Polynomial(valid_polynomial)
    # Assert indeterminates are correct.
    assert current_polynomial.indeterminates == exp_indeterminates
    # Assert monomials are correct.
    assert current_polynomial.monomials == exp_monomials
