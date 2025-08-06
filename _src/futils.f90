module types
  implicit none
  integer, parameter :: dp = kind(0.d0)
  ! integer, parameter :: dp = selected_real_kind(15, 307)

end module types

module linalg
  use types, only: dp
  implicit none

  contains

  subroutine dgem(n, A, b, x, singular)
    ! ------------------------------------------------------------------------ !
    ! Naive implementation of the Gaussian Elimination Method with partial
    ! pivoting to solve a system of linear equations Ax = b. No copies are
    ! made, meaning that the matrix A and vector b will be modified in place.
    !
    ! Parameters
    ! ----------
    ! n        : Size of input arrays.
    !
    ! Updates
    ! -------
    ! A        : Rank-two array of double precision items with shape (n, n).
    ! b        : Rank-one array of double precision items with shape (n,).
    ! x        : Rank-one array of double precision items with shape (n,).
    ! singular : A boolean flag indicating if the input matrix A is
    !            singular.
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: n
    ! f2py real(8), intent(in, out) :: A(n, n), b(n)
    ! f2py real(8), intent(out) :: x(n)
    ! f2py logical, intent(out) :: singular
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: n
    real(kind=dp), intent(inout) :: A(n, n), b(n)
    real(kind=dp), intent(out) :: x(n)
    logical, intent(out) :: singular
    integer :: j, k, prow
    real(kind=dp) :: buf
    real(kind=dp), parameter :: eps = 1.e-13_dp
    singular = .false.
    do k = 1, n - 1
      prow = maxloc(abs(A(k:n, k)), 1) + k - 1
      if (abs(A(prow, k)) <= eps) then
        singular = .true.
        return
      end if
      if (prow /= k) then
        do j = k, n
          buf = A(prow, j)
          A(prow, j) = A(k, j)
          A(k, j) = buf
        end do
        buf = b(prow)
        b(prow) = b(k)
        b(k) = buf
      end if
      A(k+1:n, k) = A(k+1:n, k) / A(k, k)
      do j = k + 1, n
        A(k+1:n, j) = A(k+1:n, j) - A(k, j) * A(k+1:n, k)
      end do
      ! A(k+1:n, k) = 0._dp
      b(k+1:n) = b(k+1:n) - A(k+1:n, k) * b(k)
    end do
    do k = n, 1, -1
      x(k) = (b(k) - dot_product(A(k, k+1:n), x(k+1:n))) / A(k, k)
    end do
  end subroutine dgem

  subroutine drqi(n, Q, x, lmbd, singular, maxiter, tol)
    ! ------------------------------------------------------------------------ !
    ! Rayleigh quotient iteration algorithm to find an eigenvalue and
    ! corresponding eigenvector close to given initial guesses.
    !
    ! Parameters
    ! ----------
    ! n        : Size of input arrays.
    ! Q        : Rank-two array of double precision items with shape (n, n).
    !            It should be symmetric.
    ! maxiter  : The maximum number of iterations.
    ! tol      : Terminate successfully if the norm of the error vector is
    !            less than tol.
    !
    ! Updates
    ! -------
    ! x        : Rank-one array of double precision items with shape (n,).
    !            It should contain initial guess of the eigenvector.
    ! lmbd     : Initial guess of the eigenvalue.
    ! singular : A boolean flag indicating if the input matrix A is
    !            singular.
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: n
    ! f2py real(8), intent(in) :: Q(n, n), tol
    ! f2py integer, intent(in) :: maxiter
    ! f2py real(8), intent(in, out) :: x(n), lmbd
    ! f2py logical, intent(out) :: singular
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: n, maxiter
    real(kind=dp), intent(in) :: tol, Q(n, n)
    real(kind=dp), intent(inout) :: x(n)
    real(kind=dp), intent(inout) :: lmbd
    logical, intent(out) :: singular
    integer :: i, k
    real(kind=dp) :: lmbdkp1, dlmbd, M(n, n), xkp1(n)
    k = 0
    xkp1 = matmul(Q, x) - lmbd * x
    if (sqrt(dot_product(xkp1, xkp1)) < tol) then
      return
    end if
    singular = .false.
    M = Q
    do i = 1, n
      M(i, i) = Q(i, i) - lmbd
    end do
    call dgem(n, M, x, xkp1, singular)
    if (singular) then
      return
    end if
    xkp1 = xkp1 / sqrt(dot_product(xkp1, xkp1))
    lmbdkp1 = dot_product(matmul(Q, xkp1), xkp1)
    dlmbd = abs((lmbdkp1 - lmbd) / lmbd)
    do while ((dlmbd > tol) .and. (k < maxiter))
      x = xkp1
      lmbd = lmbdkp1
      M = Q
      do i = 1, n
        M(i, i) = Q(i, i) - lmbd
      end do
      call dgem(n, M, x, xkp1, singular)
      if (singular) then
        return
      end if
      xkp1 = xkp1 / sqrt(dot_product(xkp1, xkp1))
      lmbdkp1 = dot_product(matmul(Q, xkp1), xkp1)
      dlmbd = abs((lmbdkp1 - lmbd) / lmbd)
      k = k + 1
    end do
    lmbd = lmbdkp1
    x = xkp1
  end subroutine drqi

  subroutine dplu(n, A, p, singular)
    ! ------------------------------------------------------------------------ !
    ! Naive implementation of the LU-decomposition with partial pivoting. No
    ! copies are made, meaning that the matrix A will be overwritten by the
    ! matrices L and U.
    !
    ! Parameters
    ! ----------
    ! n        : Size of input arrays.
    !
    ! Updates
    ! -------
    ! A        : Rank-two array of double precision items with shape (n, n).
    ! p        : Rank-one array of integer items with shape (n,). It
    !            represents the permuted rows of the matrix A.
    ! singular : A boolean flag indicating if the input matrix A is
    !            singular.
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: n
    ! f2py real(8), intent(in, out) :: A(n, n)
    ! f2py integer, intent(out) :: p(n)
    ! f2py logical, intent(out) :: singular
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: n
    real(kind=dp), intent(inout) :: A(n, n)
    integer, intent(out) :: p(n)
    logical, intent(out) :: singular
    integer :: j, k, prow
    real(kind=dp) :: buf
    real(kind=dp), parameter :: eps = 1.e-13_dp
    singular = .false.
    do k = 1, n
      p(k) = k
    end do
    do k = 1, n
      prow = maxloc(abs(A(k:n, k)), 1) + k - 1
      if (abs(A(prow, k)) <= eps) then
        singular = .true.
        return
      end if
      if (prow /= k) then
        do j = 1, n
          buf = A(prow, j)
          A(prow, j) = A(k, j)
          A(k, j) = buf
        end do
        j = p(k)
        p(k) = p(prow)
        p(prow) = j
      end if
      A(k+1:n, k) = A(k+1:n, k) / A(k, k)
      do j = k + 1, n
        A(k+1:n, j) = A(k+1:n, j) - A(k+1:n, k) * A(k, j)
      end do
    end do
  end subroutine dplu

  ! subroutine dchol()

  ! end subroutine dchol

  ! subroutine dmchol()

  ! end subroutine dmchol

end module linalg
