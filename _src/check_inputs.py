import numpy as np
import numpy.typing as npt


def check_PTyi(
  P: np.float64,
  T: np.float64,
  yi: npt.NDArray[np.float64],
  Nc: int,
  allow_negative_yi: bool = False,
) -> None:
  if not all(map(lambda a: isinstance(a, np.float64), (P, T))):
    raise TypeError(
      "Type of P and T must be numpy.float64, but"
      f"\n\t{type(P)=},\n\t{type(T)=}."
    )
  if not isinstance(yi, np.ndarray):
    raise TypeError(
      "Type of the mole fraction array (yi) must be numpy.ndarray, but"
      f"\n\t{type(yi)=}."
    )
  if not all(map(lambda a: np.issubdtype(a.dtype, np.float64), (P, T, yi))):
    raise TypeError(
      f"Data types of all input arguments must be numpy.float64, but"
      f"\n\t{P.dtype=},\n\t{T.dtype=},\n\t{yi.dtype=}."
    )
  if not yi.shape == (Nc,):
    raise ValueError(
      f"The shape of yi must be equal to ({Nc},), but {yi.shape=}."
    )
  if not np.isfinite(P):
    raise ValueError(
      f"P must be finite, but {P=}."
    )
  if not np.isfinite(T):
    raise ValueError(
      f"T must be finite, but {T=}."
    )
  if not np.isfinite(yi).all():
    where = np.where(1 - np.isfinite(yi))
    raise ValueError(
      "All mole fractions must be finite, but"
      f"\n\tyi have values: {yi[where]}\n\tat indices {where}."
    )
  if not (P > 0.):
    raise ValueError(
      "Pressure (P) must be greater than zero."
    )
  if not (T > 0.):
    raise ValueError(
      "Temperature (T) must be greater than zero."
    )
  if (yi < 0.).any() & (not allow_negative_yi):
    where = np.where(yi < 0.)
    raise ValueError(
      "All mole fractions must be greater than zero, but"
      f"\n\tyi have values: {yi[where]}\n\tat indices: {where}."
    )
  pass
