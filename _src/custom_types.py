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

  def get_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    check_input: bool = True,
  ) -> ScalarType: ...

  def get_lnphii(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    check_input: bool = True,
  ) -> VectorType: ...

  def get_lnphii_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    check_input: bool = True,
  ) -> tuple[VectorType, ScalarType]: ...

  def get_lnphiji_Zj(
    self,
    P: ScalarType,
    T: ScalarType,
    yji: MatrixType,
    check_input: bool = True,
  ) -> tuple[MatrixType, VectorType]: ...

  def get_kvguess(
    self,
    P: ScalarType,
    T: ScalarType,
    # level: int,
  ) -> MatrixType: ...

