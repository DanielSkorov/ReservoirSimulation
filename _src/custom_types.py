import numpy as np

from typing import (
  Protocol,
  TypeVar,
)

DType = TypeVar('DType', bound=np.float64)

ScalarType = np.float64

VectorType = np.ndarray[tuple[int], np.dtype[DType]]

MatrixType = np.ndarray[tuple[int, int], np.dtype[DType]]


class EOSPTType(Protocol):
  mwi: VectorType

  def getPT_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> ScalarType: ...

  def getPT_lnphii(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType: ...

  def getPT_lnphii_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType]: ...

  def getPT_lnphiji_Zj(
    self,
    P: ScalarType,
    T: ScalarType,
    yji: MatrixType,
  ) -> tuple[MatrixType, VectorType]: ...

  def getPT_kvguess(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    level: int,
  ) -> MatrixType: ...


class EOSVTType(Protocol):
  mwi: VectorType

  def getVT_P(
    self,
    V: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> ScalarType: ...

  def getVT_lnfi_dnj(
    self,
    V: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType,
  ) -> tuple[VectorType, MatrixType]: ...

  def getVT_d3F(
    self,
    V: ScalarType,
    T: ScalarType,
    yi: VectorType,
    zti: VectorType,
    n: ScalarType,
  ) -> ScalarType: ...

  def detVT_vmin(
    self,
    T: ScalarType,
    yi: VectorType,
  ) -> ScalarType: ...

