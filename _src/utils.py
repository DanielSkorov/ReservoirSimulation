import numpy as np

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  LinAlgError,
)

from futils import (
  linalg as fla,
)


def dgem(A: Matrix, b: Vector, inplace: bool = False) -> Vector:
  assert A.shape[0] == A.shape[1] == b.shape[0]
  fA = np.asfortranarray(A)
  fb = np.asfortranarray(b)
  if inplace:
    x, singular = fla.dgem_(fA, fb)
  else:
    x, singular = fla.dgem(fA, fb)
  if singular:
    raise LinAlgError('Singular matrix.')
  return x


def drqi(
  Q: Matrix,
  x0: Vector | None = None,
  lmbd0: Scalar | None = None,
  tol: Scalar = 1e-8,
  maxiter: int = 20,
) -> tuple[Vector, Scalar]:
  assert Q.shape[0] == Q.shape[1]
  fQ = np.asfortranarray(Q)
  if x0 is None:
    x = np.full((Q.shape[0],), 1. / Q.shape[0])
  else:
    assert x0.shape[0] == Q.shape[0]
    x = x0
  if lmbd0 is None:
    lmbd = np.atleast_1d(x.dot(fQ).dot(x))
  else:
    lmbd = np.atleast_1d(lmbd0)
  singular = fla.drqi(fQ, maxiter, tol, x, lmbd)
  if singular:
    raise LinAlgError('Singular matrix.')
  return x, lmbd[0]


def pyrqi(
  Q: Matrix,
  x0: Vector | None = None,
  lmbd0: Scalar | None = None,
  tol: Scalar = 1e-8,
  maxiter: int = 20,
) -> tuple[Vector, Scalar]:
  if x0 is None:
    xk = np.ones(shape=(Q.shape[0],))
  else:
    xk = x0
  if lmbd0 is None:
    lmbdk = xk.dot(Q).dot(xk)
  else:
    lmbdk = lmbd0
  M = Q - np.diagflat(np.full_like(xk, lmbdk))
  xkp1 = np.linalg.solve(M, xk)
  xkp1 /= np.linalg.norm(xkp1)
  lmbdkp1 = xkp1.dot(Q).dot(xkp1)
  k = 1
  while (np.abs((lmbdkp1 - lmbdk) / lmbdk) > tol) & (k < maxiter):
    xk = xkp1
    lmbdk = lmbdkp1
    M = Q - np.diagflat(np.full_like(xk, lmbdk))
    xkp1 = np.linalg.solve(M, xk)
    xkp1 /= np.linalg.norm(xkp1)
    lmbdkp1 = xkp1.dot(Q).dot(xkp1)
    k += 1
  return xkp1, lmbdkp1

