from typing import (
  Protocol,
)

from numpy import (
  typing as npt,
  float64,
)


class EOS_PT(Protocol):

  def get_Z(
    self,
    P: float64,
    T: float64,
    yi: npt.NDArray[float64],
    check_input: bool = True,
  ) -> float64: ...

  def get_lnphii(
    self,
    P: float64,
    T: float64,
    yi: npt.NDArray[float64],
    check_input: bool = True,
  ) -> npt.NDArray[float64]: ...

  def get_lnphii_Z(
    self,
    P: float64,
    T: float64,
    yi: npt.NDArray[float64],
    check_input: bool = True,
  ) -> tuple[npt.NDArray[float64], float64]: ...

  def get_lnphiji_Zj(
    self,
    P: float64,
    T: float64,
    yji: npt.NDArray[float64],
    check_input: bool = True,
  ) -> tuple[npt.NDArray[float64], npt.NDArray[float64]]: ...

  def get_kvguess(
    self,
    P: float64,
    T: float64,
    # level: int,
  ) -> npt.NDArray[float64]: ...

