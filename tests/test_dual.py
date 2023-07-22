from team48_autodiff_package.dualNumber.dual_numbers import *
import math
import pytest

def test_init():
    d1 = DualNumber(2)
    d2 = DualNumber(7, 3)

    assert d1.real == 2
    assert d1.dual == 1
    assert d2.real == 7
    assert d2.dual == 3

def test_neg():
    d1 = DualNumber(5)
    d2 = DualNumber(6, 3)

    assert -d1 == DualNumber(-5, -1)
    assert -d2 == DualNumber(-6, -3)

def test_add():
    d1 = DualNumber(7, 8)
    d2 = DualNumber(50, 700)

    assert d1 + 5 == DualNumber(12, 8)
    assert d2 + 7 == DualNumber(57, 700)
    assert d1 + d2 == DualNumber(57, 708)
    assert d2 + d1 == DualNumber(57, 708)
    with pytest.raises(TypeError):
        d1 + "hi"

def test_radd():
    d1 = DualNumber(7, 8)
    d2 = DualNumber(50, 700)

    assert 5 + d1 == DualNumber(12, 8)
    assert 7 + d2 == DualNumber(57, 700)

def test_sub():
    d1 = DualNumber(80, 90)
    d2 = DualNumber(1000, 4)

    assert d1 - 1 == DualNumber(79, 90)
    assert d2 - 700 == DualNumber(300, 4)
    assert d1 - d2 == DualNumber(-920, 86)
    assert d2 - d1 == DualNumber(920, -86)
    with pytest.raises(TypeError):
        d1 - "hi"

def test_rsub():
    d1 = DualNumber(80, 90)
    d2 = DualNumber(1000,4)

    assert 100 - d1 == DualNumber(20, -90)
    assert 1400 - d2 == DualNumber(400, -4)


def test_mul():
    d1 = DualNumber(5)
    d2 = DualNumber(19, 2)

    assert d1 * 5 == DualNumber(25, 5)
    assert d2 * 3 == DualNumber(57, 6)
    assert d1 * d2 == DualNumber(95, 29)
    assert d2 * d1 == DualNumber(95, 29)
    with pytest.raises(TypeError):
        d1 * "hi"

def test_rmul():
    d1 = DualNumber(5)
    d2 = DualNumber(19,2)

    assert 5 * d1 == DualNumber(25, 5)
    assert 3 * d2 == DualNumber (57, 6)

def test_div():
    d1 = DualNumber(18, 3)
    d2 = DualNumber(8, 2.0)
    d3 = DualNumber(0, 0)

    assert d1 / 3 == DualNumber(6, 1)
    assert d1 / d2 == DualNumber(18/8, (8 * 3 - 18 * 2) / (8 * 8))
    assert d2 / 4 == DualNumber(2, 0.5)
    assert d2 / d1 == DualNumber(8/18, (18 * 2 - 8 * 3) / (18 * 18))
    with pytest.raises(ZeroDivisionError):
        d1 / 0
    with pytest.raises(ZeroDivisionError):
        d1 / d3
    with pytest.raises(ZeroDivisionError):
        d1 / 0.0
    with pytest.raises(TypeError):
        d1 / "hi"


def test_rdiv():
    d1 = DualNumber(3, 3)
    d2 = DualNumber(2, 5)
    d3 = DualNumber(0, 0)

    assert 9 / d1 == DualNumber(3, -3)
    assert 4 / d2 == DualNumber(2, -5)
    with pytest.raises(ZeroDivisionError):
        4 / d3

def test_pow():
    d1 = DualNumber(6, 9)
    d2 = DualNumber(4,2)

    assert d1 ** 2 == DualNumber(36, 108)
    assert d2 ** 3 == DualNumber(64, 96)
    assert d2 ** d1 == DualNumber(4096, 4096 * (9 * math.log(4) + 3))
    assert d1 ** d2 == DualNumber(1296, 1296 * (2 * math.log(6) + 6))
    with pytest.raises(TypeError):
        d1**"hi"

def test_rpow():
    d1 = DualNumber(6, 9)
    d2 = DualNumber(4,2)

    assert 2 ** d1 == DualNumber(64, 576 * math.log(2))
    assert 3 ** d2 == DualNumber(81, 162 * math.log (3))

def test_sin():
    d1 = DualNumber(5)
    d2 = DualNumber(7, 8)

    assert sin(d1) == DualNumber(np.sin(5), np.cos(5))
    assert sin(d2) == DualNumber(np.sin(7), 8 * np.cos(7))
    assert sin(11) == np.sin(11)
    with pytest.raises(TypeError):
        sin("hi")

def test_cos():
    d1 = DualNumber(4, 88)
    d2 = DualNumber (9, 3)

    assert cos(d1) == DualNumber(np.cos(4), -88 * np.sin(4))
    assert cos(d2) == DualNumber(np.cos(9), -3 * np.sin(9))
    assert cos(7) == np.cos(7)
    with pytest.raises(TypeError):
        cos("hi")

def test_tan():
    d1 = DualNumber(45, 6)
    d2 = DualNumber(3)

    assert tan(d1) == DualNumber(np.tan(45), 6 / (np.cos(45))**2)
    assert tan(d2) == DualNumber(np.tan(3), 1 / np.cos(3) ** 2)
    assert tan(5) == np.tan(5)
    with pytest.raises(TypeError):
        tan("hi")

def test_log():
    d1 = DualNumber(2, 3)
    d2 = DualNumber(50)

    assert log(d1) == DualNumber(math.log(2), 1.5)
    assert log(d2) == DualNumber(math.log(50), .02)
    assert log(d1, 2) == DualNumber(math.log(d1.real, 2), d1.dual / (d1.real * math.log(2)))
    assert log(d2, 4) == DualNumber(math.log(d2.real, 4), d2.dual / (d2.real * math.log(4)))
    assert log(math.e) == 1
    with pytest.raises(TypeError):
        log("hi")
    with pytest.raises(ValueError):
        log(DualNumber(-5, -6))

def test_exp():
    d1 = DualNumber(20, 5)
    d2 = DualNumber(7)

    assert exp(d1) == DualNumber(np.e**d1.real, np.e**(d1.real) * d1.dual)
    assert exp(d2) == DualNumber(np.e**(d2.real), np.e**(d2.real) * d2.dual)
    with pytest.raises(TypeError):
        exp("hi")

def test_sqrt():
    d1 = DualNumber(16, 4)
    d2 = DualNumber(9, 15)

    assert sqrt(d1) == DualNumber(4, 0.5)
    assert sqrt(d2) == DualNumber(3, 15 / 6)
    assert sqrt(25) == 5
    with pytest.raises(TypeError):
        sqrt("hi")

def test_eq():
    d1 = DualNumber(10, 2)
    d2 = DualNumber(20, 2)
    d3 = DualNumber(10, 3)
    d4 = DualNumber(10,2)

    assert (d1.real == d3.real and d1.dual == d3.dual) == (d1 == d3)
    assert (d1.real == d2.real and d1.dual == d2.dual) == (d1 == d2)
    assert (d1.real == d4.real and d1.dual == d4.dual) == (d1 == d4)
    assert (d2.real == d3.real and d2.dual == d3.dual) == (d2 == d3)

def test_asin():
    d1 = DualNumber(.5, .75)
    d2 = DualNumber(.3, .67)
    d3 = DualNumber(2, 3)

    assert asin(d1) == DualNumber(np.arcsin(d1.real), d1.dual / math.sqrt(1 - d1.real**2))
    assert asin(d2) == DualNumber(np.arcsin(d2.real), d2.dual / math.sqrt(1 - d2.real**2))
    with pytest.raises(TypeError):
        asin("hi")
    with pytest.raises(ValueError):
        asin(d3)

def test_acos():
    d1 = DualNumber(.55, .7)
    d2 = DualNumber(-.2, .62)
    d3 = DualNumber(5, 1)

    assert acos(d1) == DualNumber(np.arccos(d1.real), d1.dual / -math.sqrt(1 - d1.real**2))
    assert acos(d2) == DualNumber(np.arccos(d2.real), d2.dual / -math.sqrt(1 - d2.real**2))
    with pytest.raises(TypeError):
        acos("hi")
    with pytest.raises(ValueError):
        acos(d3)

def test_atan():
    d1 = DualNumber(32, 88)
    d2 = DualNumber(78, 9)
    d3 = DualNumber(5, 6)

    assert atan(d1) == DualNumber(math.atan(d1.real), d1.dual / (1 + d1.real**2))
    assert atan(d2) == DualNumber(math.atan(d2.real), d2.dual / (1 + d2.real**2))
    assert atan(d3) == DualNumber(math.atan(d3.real), d3.dual / (1 + d3.real**2))

    with pytest.raises(TypeError):
        atan("hi")

def test_sinh():
    d1 = DualNumber(.5, .75)
    d2 = DualNumber(.3, .67)

    assert sinh(d1) == DualNumber(math.sinh(d1.real), d1.dual * math.cosh(d1.real))
    assert sinh(d2) == DualNumber(math.sinh(d2.real), d2.dual * math.cosh(d2.real))
    with pytest.raises(TypeError):
        sinh("hi")

def test_cosh():
    d1 = DualNumber(.55, .7)
    d2 = DualNumber(-.2, 62)

    assert cosh(d1) == DualNumber(math.cosh(d1.real), d1.dual * math.sinh(d1.real))
    assert cosh(d2) == DualNumber(math.cosh(d2.real), d2.dual * math.sinh(d2.real))
    with pytest.raises(TypeError):
        cosh("hi")

def test_tanh():
    d1 = DualNumber(32, 88)
    d2 = DualNumber(78, 9)
    d3 = DualNumber(5, 6)

    assert tanh(d1) == sinh(d1) / cosh(d1)
    assert tanh(d2) == sinh(d2) / cosh(d2)
    assert tanh(d3) == sinh(d3) / cosh(d3)

    with pytest.raises(TypeError):
        sinh("hi")

def test_powexp():
    d1 = DualNumber(6, 9)
    d2 = DualNumber(4,2)

    assert pow(d1, 2)== DualNumber(36, 108)
    assert pow(d2, 3)== DualNumber(64, 96)
    assert pow(d2, d1) == DualNumber(4096, 4096 * (9 * math.log(4) + 3))
    assert pow(d1, d2) == DualNumber(1296, 1296 * (2 * math.log(6) + 6))
    with pytest.raises(TypeError):
        pow(d1, "hi")
    with pytest.raises(TypeError):
        pow("hi", d1)

def test_logistic():
    d1 = DualNumber(5, 3)
    d2 = DualNumber(3)

    assert logistic(d1) == 1 / (1 + exp(-d1))
    assert logistic(d1, 2, 3, 4) == 2 / (1 + exp(-3*(d1 - 4)))
    assert logistic(d2, 4) == 4 / (1 + exp(-d2))
    assert logistic(d2) == 1 / (1 + exp(-d2))
    with pytest.raises(TypeError):
        logistic("hi")
    with pytest.raises(TypeError):
        logistic(d1, "hi")
    with pytest.raises(TypeError):
        logistic(d1, 1, "hi")
    with pytest.raises(TypeError):
        logistic(d1, 2, 3, "hi")