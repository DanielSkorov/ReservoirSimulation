import numpy as np

from typing import (
  Protocol,
  TypeVar,
)


DType = TypeVar('DType', bound=np.float64)

Scalar = np.float64 | float

Vector = np.ndarray[tuple[int], np.dtype[DType]]

Matrix = np.ndarray[tuple[int, int], np.dtype[DType]]

Tensor = np.ndarray[tuple[int, int, int], np.dtype[DType]]


class Eos(Protocol):
  name: str
  Nc: int
  mwi: Vector


class SolutionNotFoundError(Exception):
  pass

class LinAlgError(Exception):
  pass
