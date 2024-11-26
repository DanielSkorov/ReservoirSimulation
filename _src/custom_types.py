import numpy as np

from typing import (
  Protocol,
  TypeVar,
)

DType = TypeVar('DType', bound=np.float64)

ScalarType = np.float64

VectorType = np.ndarray[tuple[int], np.dtype[DType]]

MatrixType = np.ndarray[tuple[int, int], np.dtype[DType]]


class EosptType(Protocol):
  mwi: VectorType

  def get_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> ScalarType: ...

  def get_lnphii(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType: ...

  def get_lnphii_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType]: ...

  def get_lnphiji_Zj(
    self,
    P: ScalarType,
    T: ScalarType,
    yji: MatrixType,
  ) -> tuple[MatrixType, VectorType]: ...

  def get_kvguess(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    level: int,
  ) -> MatrixType: ...

