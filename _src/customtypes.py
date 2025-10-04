import numpy as np

from typing import (
  TypeVar,
  TypeAlias,
  Callable,
)


Logical = np.bool

Integer = np.int_

Double = np.float64


T = TypeVar('T', bound=np.generic)


Vector: TypeAlias = np.ndarray[tuple[int], np.dtype[T]]

Matrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[T]]

Tensor: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[T]]


# Linsolver = Callable[[Matrix[Double], Vector[Double]], Vector[Double]]

Linsolver = Callable


class SolutionNotFoundError(Exception):
  pass

class LinAlgError(Exception):
  pass
