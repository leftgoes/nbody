import numpy as np

T = np.ndarray[float] | float


def pareto(x: T, m: float, alpha: float) -> T:
    return m/(1 - x)**(1/alpha)


def exponential(x: T, m: float, alpha: float) -> T:
    return m - np.log(1 - x)/alpha


def mass(n: int, zero: float, half: float) -> T:
    alpha = -np.log(2) / np.log(zero/half)
    return pareto(np.random.random(n), zero, alpha)


def radius(n: int, zero: float, half: float) -> T:
    alpha = np.log(2)/(half - zero)
    return exponential(np.random.random(n), zero, alpha)


def angle(n: int, zero: float = 0, one: float = 2 * np.pi) -> T:
    return np.random.random(n) * (one - zero) + zero