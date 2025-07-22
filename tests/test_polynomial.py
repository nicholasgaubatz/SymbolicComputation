"""Tests for the polynomial submodule."""

import re

import numpy as np
import pytest

from symboliccomputation.indeterminate import Indeterminate
from symboliccomputation.polynomial import Monomial, Polynomial

########################################################################################
# Monomials
########################################################################################


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
    "-",
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


@pytest.mark.parametrize(argnames=("monomial_1", "monomial_2", "exp_monomial_prod"),
                         argvalues=[
    (Monomial("0"), Monomial("0.0"), Monomial("0")),
    (Monomial("0"), Monomial("2"), Monomial("0.0")),
    (Monomial("x"), Monomial("0.0"), Monomial("0")),
    (Monomial("x"), Monomial("y"), Monomial("x*y")),
    (Monomial("x"), Monomial("2*y"), Monomial("2*x*y")),
    (Monomial("-3.0*asdf"), Monomial("9*asdf"), Monomial("-27*asdf^2")),
])
def test_monomial_mul(monomial_1: list[Monomial],
                      monomial_2: list[Monomial],
                      exp_monomial_prod: list[Monomial]) -> None:
    """Tests some Monomial multiplications."""
    assert monomial_1 * monomial_2 == exp_monomial_prod


@pytest.mark.parametrize(argnames=("monomial", "wrt", "result"), argvalues=[
    (Monomial("0"), "x", Monomial("0.0")),
    (Monomial("1"), Indeterminate("x"), Monomial("0")),
    (Monomial("x"), "x", Monomial("1")),
    (Monomial("x"), "y", Monomial("0")),
    (Monomial("2*asdf"), "asdf", Monomial("2")),
    (Monomial("-4*x*y"), "x", Monomial("-4*y")),
    (Monomial("4*x^2*y"), "x", Monomial("8.0*x*y")),
])
def test_monomial_derivative(monomial: list[Monomial],
                             wrt: list[str | Indeterminate],
                             result: list[Monomial]) -> None:
    """Tests some Monomial derivatives."""
    assert monomial.derivative(wrt) == result


########################################################################################
# Polynomials
########################################################################################


@pytest.mark.parametrize(argnames=("invalid_polynomial", "exp_error_msg"), argvalues=[
    ("asdf1", "Invalid input: token 'asdf1'!"),
    ("1.", "Invalid input: token '1.'!"),
    ("x^", "Invalid input: token 'x^'!"),
    ("x2", "Invalid input: token 'x2'!"),
    ("x^(1)", "Invalid input: token 'x^(1)'!"),
    ("x^-1", "Invalid input: token 'x^-1'!"),
    ("x^0", "Invalid input: token 'x^0'!"),
    ("x ++ y", "Invalid input: token '++'!"),
    ("x y", "Invalid sequential tokens: x, y!"),
    ("x + x", "Invalid input: repeated monomials x, x!"),
    ("x + 1.0*x", "Invalid input: repeated monomials x, x!"),
    ("x + y + x*y + x*y*z - 12345*x",
     "Invalid input: repeated monomials x, -12345.0*x!"),
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
     ("3*x + 4.0*x*y - 8*y + 12",
      {Indeterminate("x"), Indeterminate("y")},
      {Monomial("3*x"), Monomial("4.0*x*y"), Monomial("-8*y"), Monomial("12")}),
])
def test_valid_polynomial(valid_polynomial: list[str],
                        exp_indeterminates: set[Indeterminate],
                        exp_monomials: set[Monomial]) -> None:
    """Tests some valid monomial regular expressions."""
    current_polynomial = Polynomial(valid_polynomial)
    # Assert indeterminates are correct.
    assert current_polynomial.indeterminates == exp_indeterminates
    # Assert monomials are correct.
    assert current_polynomial.monomials == exp_monomials


@pytest.mark.parametrize(argnames=("polynomial_1",
                                   "polynomial_2",
                                   "exp_polynomial_sum"),
                         argvalues=[
    ("0", "0.0", Polynomial("0")),
    ("1.0", "1", Polynomial("2")),
    ("-2.5", "2", Polynomial("-0.5")),
    ("x", "y", Polynomial("x + y")),
    ("x", "-1*y", Polynomial("x - y")),
    ("x", "-1*x", Polynomial("0")),
    ("2*x^2 + x*y",
     "3*x^3*y - 2*x^2 + 2.1*x*y - 8",
     Polynomial("3.1*x*y + 3*x^3*y - 8")),
])
def test_polynomial_sum(polynomial_1: list[str],
                        polynomial_2: list[str],
                        exp_polynomial_sum: list[Polynomial]) -> None:
    """Tests sums of polynomials."""
    assert Polynomial(polynomial_1) + Polynomial(polynomial_2) == exp_polynomial_sum


@pytest.mark.parametrize(argnames=("polynomial_1",
                                   "polynomial_2",
                                   "exp_polynomial_diff"),
                         argvalues=[
    ("0", "0.0", Polynomial("0")),
    ("1.0", "1", Polynomial("0")),
    ("-2.5", "2", Polynomial("-4.5")),
    ("x", "y", Polynomial("x - y")),
    ("x", "-1*y", Polynomial("x + y")),
    ("x", "-1*x", Polynomial("2*x")),
    ("x", "1.0*x", Polynomial("0")),
    ("2*x^2 + x*y",
     "3*x^3*y + 2*x^2 + 2.1*x*y - 8",
     Polynomial("-1.1*x*y + -3*x^3*y + 8")),
])
def test_polynomial_diff(polynomial_1: list[str],
                        polynomial_2: list[str],
                        exp_polynomial_diff: list[Polynomial]) -> None:
    """Tests differences of polynomials."""
    assert Polynomial(polynomial_1) - Polynomial(polynomial_2) == exp_polynomial_diff


@pytest.mark.parametrize(argnames=("polynomial_1",
                                   "polynomial_2",
                                   "exp_polynomial_prod"),
                         argvalues=[
    (Polynomial("0"), Polynomial("0.0"), Polynomial("0")),
    (Polynomial("0"), Polynomial("2"), Polynomial("0.0")),
    (Polynomial("x"), Polynomial("0.0"), Polynomial("0")),
    (Polynomial("x"), Polynomial("y"), Polynomial("x*y")),
    (Polynomial("x"), Polynomial("2*y"), Polynomial("2*x*y")),
    (Polynomial("-3.0*asdf"), Polynomial("9*asdf"), Polynomial("-27*asdf^2")),
    (Polynomial("x + y"), Polynomial("x - y"), Polynomial("x^2 - y^2")),
    (Polynomial("x + y"), Polynomial("x + y"), Polynomial("x^2 + 2*x*y + y^2")),
    (0, Polynomial("x + y"), Polynomial("0")),
    (1, Polynomial("-3.0*asdf"), Polynomial("-3*asdf")),
    (1, Polynomial("-3.0*asdf"), Polynomial("-3.0*asdf")),
    (-2, Polynomial("8*x + 9*y - 10*x*y"), Polynomial("-16.0*x - 18*y + 20*x*y")),
])
def test_polynomial_mul(polynomial_1: list[Polynomial | int | float],
                      polynomial_2: list[Polynomial],
                      exp_polynomial_prod: list[Polynomial]) -> None:
    """Tests some Polynomial multiplications and left scalar Polynomial mults."""
    assert polynomial_1 * polynomial_2 == exp_polynomial_prod


@pytest.mark.parametrize(argnames=("polynomial", "wrt", "result"), argvalues=[
    (Polynomial("0"), "x", Polynomial("0.0")),
    (Polynomial("1"), Indeterminate("x"), Polynomial("0")),
    (Polynomial("x"), "x", Polynomial("1")),
    (Polynomial("x"), "y", Polynomial("0")),
    (Polynomial("2*asdf"), "asdf", Polynomial("2")),
    (Polynomial("-4*x*y"), "x", Polynomial("-4*y")),
    (Polynomial("4*x^2*y"), "x", Polynomial("8.0*x*y")),
    (Polynomial("x + y"), "x", Polynomial("1")),
    (Polynomial("-1*x + y"), "x", Polynomial("-1")),
    (Polynomial("x*y + x + y"), "z", Polynomial("0")),
    (Polynomial("4*wow - 8*wow^2 + 12345*wow^3"),
     "wow",
     Polynomial("37035*wow^2 - 16*wow + 4.0")),
    (Polynomial("4*wow - 8*wow^2 + 12345*wow^3"), "wowza", Polynomial("0")),
])
def test_polynomial_derivative(polynomial: list[Polynomial],
                             wrt: list[str | Indeterminate],
                             result: list[Polynomial]) -> None:
    """Tests some Polynomial derivatives."""
    assert polynomial.derivative(wrt) == result
