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
    ! Solves the system of linear equations Ax = b by the Gaussian
    ! elimination method with partial pivoting.
    !
    ! Parameters
    ! ----------
    ! n        : Size of input arrays.
    ! A        : Rank-two array of double precision items with shape (n, n).
    ! b        : Rank-one array of double precision items with shape (n,).
    !
    ! Updates
    ! -------
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
    integer :: i, j, k, pivot_row
    real(kind=dp) :: pivot, summ, element
    real(kind=dp), parameter :: eps = 1.e-13_dp
    singular = .false.
    do k = 1, n - 1
      pivot_row = maxval(maxloc(abs(A(k:n, k)))) + k - 1
      if (abs(A(pivot_row, k)) <= eps) then
        singular = .true.
        return
      end if
      if (pivot_row .ne. k) then
        element = A(pivot_row, k)
        A(pivot_row, k) = A(k, k)
        A(k, k) = element
        element = b(pivot_row)
        b(pivot_row) = b(k)
        b(k) = element
      end if
      A(k+1:n, k) = A(k+1:n, k) / A(k, k)
      do j = k + 1, n
        pivot = A(pivot_row, j)
        if (pivot_row .ne. k) then
          A(pivot_row, j) = A(k, j)
          A(k, j) = pivot
        end if
        A(k+1:n, j) = A(k+1:n, j) - pivot * A(k+1:n, k)
      end do
      b(k+1:n) = b(k+1:n) - A(k+1:n, k) * b(k)
    end do
    do i = n, 1, -1
      summ = 0._dp
      do j = i + 1, n
        summ = summ + A(i, j) * x(j)
      end do
      x(i) = (b(i) - summ) / A(i, i)
    end do
  end subroutine dgem

  ! subroutine dchol()

  ! end subroutine dchol

  ! subroutine dmchol()

  ! end subroutine dmchol

end module linalg
