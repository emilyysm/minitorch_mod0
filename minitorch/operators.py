"""Collection of the core mathematical operators used throughout the code base."""

from math import e
from math import log as mlog
from typing import Any, Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiply 'a' by 'b'."""
    return a * b


def id(x: float) -> float:
    """Return 'x'."""
    return x


def add(a: float, b: float) -> float:
    """Add 'a' and 'b'."""
    return a + b


def neg(x: float) -> float:
    """Negate 'x'."""
    return -x


def lt(a: float, b: float) -> bool:
    """Check if 'a' < 'b'."""
    return a < b


def eq(a: float, b: float) -> bool:
    """Check if 'a' and 'b' are equal."""
    return a == b


def max(a: float, b: float) -> float:
    """Return the max between 'a' and 'b'."""
    if a <= b:
        return b
    else:
        return a


def is_close(a: float, b: float) -> bool:
    """Check if 'a' and 'b' are close in value."""
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function for 'x'."""
    if x >= 0:
        return 1.0 / (1.0 + e ** (-x))
    else:
        return e**x / (1.0 + e**x)


def relu(x: float) -> float:
    """Apply relu activation function to 'x'."""
    return max(x, 0)


def log(x: float) -> float:
    """Apply log function to 'x'."""
    return mlog(x)


def exp(x: float) -> float:
    """Calculate the exponential function, i.e., e^x."""
    return e**x


def inv(x: float) -> float:
    """Calculate the inverse of 'x'."""
    return 1 / x


def log_back(a: float, b: float) -> float:
    """Compute the derivative of ln('a') and multiply by 'b'."""
    return b / a


def inv_back(a: float, b: float) -> float:
    """Compute the derivative of the reciprocal of 'a' and multiply by 'b'."""
    return -b / (a**2)


def relu_back(a: float, b: float) -> float:
    """Compute the derivative of relu('a') and multiply by 'b'."""
    if a > 0:
        return b
    else:
        return 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], lst: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        fn: Function to apply to each element of lst.
        lst: Iterable over which to apply function fn.

    Returns:
    -------
        New iterable with function fn applied to elements.

    """
    return [fn(item) for item in lst]


def zipWith(
    fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        fn: Function to apply to each element of lst.
        lst1: First iterable over which to apply function fn.
        lst2: Second iterable over which to apply function fn.

    Returns:
    -------
        New iterable with elements from lst1 and lst2 combined using function fn.

    """
    iter1 = iter(lst1)
    iter2 = iter(lst2)
    results = []
    while True:
        try:
            elt1 = next(iter1)
            elt2 = next(iter2)
        except StopIteration:
            break
        results.append(fn(elt1, elt2))
    return results


def reduce(
    fn: Callable[[float, float], float], lst: Iterable[float], initializer: Any = None
) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        fn: Function to apply to elements in lst.
        lst: Iterable over which to apply function fn.
        initializer: Initial value to start the reduction.

    Returns:
    -------
        Single value resulting from applying function fn elements of lst.

    """
    iterator = iter(lst)
    if initializer is None:
        try:
            val = next(iterator)
        except StopIteration:
            raise ValueError("Empty list.")
    else:
        val = initializer
    for elt in iterator:
        val = fn(val, elt)
    return val


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        lst: List of values to negate.

    Returns:
    -------
        List of values from lst, negated.

    """
    return map(neg, lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        lst1: First list to add.
        lst2: Second list to add.

    Returns:
    -------
        List of values that are the sum of lst1 and lst2.

    """
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Add all elements in a list.

    Args:
    ----
        lst: List of elements to add.

    Returns:
    -------
        Sum of elements in lst.

    """
    return reduce(add, lst, 0)


def prod(lst: Iterable[float]) -> float:
    """Multiply all elements in a list."""
    return reduce(mul, lst, 1)
