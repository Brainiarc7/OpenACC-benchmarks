!!===----------------------------------------------------------------------===//
!!
!!     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
!!        compiler for NVIDIA GPUs, targeting numerical modeling code.
!!
!! This file is distributed under the University of Illinois Open Source
!! License. See LICENSE.TXT for details.
!!
!!===----------------------------------------------------------------------===//

! Generate full-length input signal X and filter mask H
! on master node and distribute them accross worker nodes.
subroutine generate_input(H, szh, X, szx, HH, szhh, XX, szxx, mszxx, mrank, msize, master)

  implicit none
  include 'mpif.h'
  integer, intent(in) :: mrank, msize, master
  integer(kind=IKIND), intent(in) :: szh, szx
  integer(kind=IKIND), intent(in) :: szhh, szxx, mszxx(0:msize-1)
  real(kind=RKIND), intent(out) :: H(szh), X(szx)
  real(kind=RKIND), intent(out) :: HH(szhh), XX(szxx)
  integer(kind=IKIND) :: i, ioffset
  real(kind=RKIND) :: roffset
  double precision :: tstart, tfinish
  integer :: ierr, mnode, hreq, xreq
  integer, parameter :: rkind = RKIND, htag = 0, xtag = 1

  if (mrank .eq. master) then
    do i = 1, szh
      call random_number(H(i))
    enddo
    do i = 1, szx
      call random_number(X(i))
    enddo
  
    ! Put an exact filter match under random offset (to check
    ! correctness).
    call random_number(roffset)
    ioffset = int(roffset * (szx - szh))
    do i = 1, szh
      X(i + ioffset) = 1.0_rkind / H(i)
    enddo
    write(*,*) 'seeded match at X(i), i = ', 1 + ioffset
  endif

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Barrier'
    stop
  endif

  if (mrank .eq. master) then
    tstart = MPI_Wtime()
  endif

  ! Send portions of the input arrays to their corresponding
  ! worker nodes.  
  if (mrank .eq. master) then
    do mnode = 0, msize - 1
      call MPI_Isend(H(1), szhh * rkind, MPI_BYTE, mnode, htag, MPI_COMM_WORLD, hreq, ierr)
      if (ierr .ne. MPI_SUCCESS) then
        write(*,*) 'Error in MPI_Isend'
        stop
      endif
    enddo

    do mnode = 0, msize - 1
      call MPI_Isend(X(1 + mnode * (szx / msize)), mszxx(mnode) * rkind, MPI_BYTE, mnode, xtag, MPI_COMM_WORLD, xreq, ierr)
      if (ierr .ne. MPI_SUCCESS) then
        write(*,*) 'Error in MPI_Isend'
        stop
      endif
    enddo
  endif
  
  ! Receive entire worker node's arrays from the master node.
  call MPI_Recv(HH(1), szhh * rkind, MPI_BYTE, master, htag, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Recv'
    stop
  endif
  call MPI_Recv(XX(1), szxx * rkind, MPI_BYTE, master, xtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Recv'
    stop
  endif  

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Barrier'
    stop
  endif

  if (mrank .eq. master) then
    tfinish = MPI_Wtime()
    write(*,*) 'data distribution time = ', tfinish - tstart, ' sec'
  endif

end subroutine generate_input



! Collect outputs from workers on the master node and
! perform simple correctness test: check output result
! contains the correct number of exact matches.
subroutine check_output(Y, szy, szh, YY, szyy, mszyy, szhh, mrank, msize, master)

  implicit none
  include 'mpif.h'
  integer, intent(in) :: mrank, msize, master
  integer(kind=IKIND), intent(in) :: szy, szh
  integer(kind=IKIND), intent(in) :: szyy, mszyy(0:msize-1), szhh
  real(kind=RKIND), intent(in) :: Y(szy)
  real(kind=RKIND), intent(in) :: YY(szyy)
  integer(kind=IKIND) :: i
  real(kind=RKIND) :: eps
  double precision :: tstart, tfinish
  integer :: ierr, mnode, yreq
  integer, parameter :: ytag = 2
  integer, dimension(0:msize-1) :: myreq
  integer, parameter :: rkind = RKIND

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Barrier'
    stop
  endif

  if (mrank .eq. master) then
    tstart = MPI_Wtime()
  endif
  
  ! Send portions of input arrays from their corresponding
  ! worker nodes to the master node.
  call MPI_Isend(YY(1), szyy * rkind, MPI_BYTE, master, ytag, MPI_COMM_WORLD, yreq, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Isend'
    stop
  endif

  if (mrank .eq. master) then
  
    ! Receive portions of input arrays from worker nodes into
    ! full-length arrays on master node.
    do mnode = 0, msize - 1
      call MPI_Irecv(Y(1 + mnode * ((szy + szh - 1) / msize)), mszyy(mnode) * rkind, &
        MPI_BYTE, mnode, ytag, MPI_COMM_WORLD, myreq(mnode), ierr)
      if (ierr .ne. MPI_SUCCESS) then
        write(*,*) 'Error in MPI_Irecv'
        stop
      endif
    enddo

    do mnode = 0, msize - 1
      call MPI_Wait(myreq(mnode), MPI_STATUS_IGNORE, ierr)
      if (ierr .ne. MPI_SUCCESS) then
        write(*,*) 'Error in MPI_Wait'
        stop
      endif
    enddo
  endif

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Barrier'
    stop
  endif

  if (mrank .eq. master) then
    tfinish = MPI_Wtime()
    write(*,*) 'results gathering time = ', tfinish - tstart, ' sec'
  endif
  
  if (mrank .eq. master) then
    eps = epsilon(eps)
    do i = 1, szy
      if (abs(Y(i) - szh) .le. eps) then
        write(*,*) 'found match at Y(i), i = ', i
      endif
    enddo
  endif

end subroutine check_output



! Perform filter matching.
subroutine match_filter(HH, szhh, XX, szxx, YY, szyy)

  implicit none
  integer(kind=IKIND), intent(in) :: szhh, szxx, szyy
  real(kind=RKIND), intent(in) :: HH(szhh), XX(szxx)
  real(kind=RKIND), intent(out) :: YY(szyy)
  integer(kind=IKIND) :: i, j
  integer, parameter :: rkind = RKIND

  do i = 1, szyy
    YY(i) = 0.0_rkind
    do j = 1, szhh
      YY(i) = YY(i) + XX(i + j - 1) * HH(j)
    enddo
  enddo

end subroutine match_filter



program main

  implicit none
  include 'mpif.h'
  character(len=128) :: arg
  integer(kind=IKIND) :: szh, szx, szy
  integer(kind=IKIND) :: szhh, szxx, szyy
  integer(kind=IKIND), allocatable, dimension(:) :: mszxx, mszyy
  real(kind=RKIND), allocatable, dimension(:) :: H, X, Y
  real(kind=RKIND), allocatable, dimension(:) :: HH, XX, YY
  double precision :: tstart, tfinish
  integer :: argc, ierr, mrank, msize, mnode
  integer, parameter :: rkind = RKIND
  integer, parameter :: master = 0

  call MPI_Init(ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Init'
    stop
  endif

  call MPI_Comm_Rank(MPI_COMM_WORLD, mrank, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Comm_Rank'
    stop
  endif

  ! Check usage
  if (mrank .eq. master) then
    argc = command_argument_count()
    if (argc .ne. 2) then
      call get_command_argument(0, arg)
      write(*,*) 'Usage: ', trim(arg), ' <signal_length> <filter_length>'
      call MPI_Finalize(ierr)
      if (ierr .ne. MPI_SUCCESS) then
        write(*,*) 'Error in MPI_Finalize'
      endif
      stop
    endif
  endif

  ! Setup szh, szx, szy
  call get_command_argument(1, arg)
  read(arg, '(I64)') szx
  call get_command_argument(2, arg)
  read(arg, '(I64)') szh
  szy = szx - (szh - 1)
  
  ! Check valid numbers
  if (szx < szh) then
    if (mrank .eq. master) then
      write(*,*) '<signal_length> must be greater or equal to the <filter_length>'
    endif
    call MPI_Finalize(ierr)
    if (ierr .ne. MPI_SUCCESS) then
      write(*,*) 'Error in MPI_Finalize'
    endif
    stop
  endif

  ! Allocate initial full-length H, X and Y arrays on the master node.
  if (mrank .eq. master) then
    allocate(H(szh), X(szx), Y(szy))
  endif

  call MPI_Comm_Size(MPI_COMM_WORLD, msize, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Comm_Size'
    stop
  endif

  szhh = szh
  szxx = szx / msize

  ! Track sizes of all arrays portions on master node.
  if (mrank .eq. master) then
    allocate(mszxx(0:msize-1), mszyy(0:msize-1))
    do mnode = 0, msize - 2
      mszxx(mnode) = szxx + (szh - 1)
      mszyy(mnode) = szxx
    enddo
    do mnode = msize - 1, msize - 1
      mszxx(mnode) = szxx + mod(szx, msize)
      mszyy(mnode) = mszxx(mnode) - (szh - 1)
    enddo
  endif

  ! Append the shadow boundary to all other nodes.    
  ! Append the remainder to the last node.
  if (mrank .ne. msize - 1) then
    szyy = szxx
    szxx = szxx + (szh - 1)
  else
    szxx = szxx + mod(szx, msize)
    szyy = szxx - (szhh - 1)
  endif

  ! Allocate portions of H, X and Y arrays distributed accross MPI nodes.
  allocate(HH(szhh), XX(szxx), YY(szyy))

  ! Generate full-length input signal X and filter mask H
  ! on master node and distribute them accross worker nodes.
  call generate_input(H, szh, X, szx, HH, szhh, XX, szxx, mszxx, mrank, msize, master)

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Barrier'
    stop
  endif

  if (mrank .eq. master) then
    tstart = MPI_Wtime()
  endif
  
  ! Perform filter matching.
  call match_filter(HH, szhh, XX, szxx, YY, szyy)

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Barrier'
    stop
  endif

  if (mrank .eq. master) then
    tfinish = MPI_Wtime()
    write(*,*) 'matching time = ', tfinish - tstart, ' sec'
  endif

  ! Collect outputs from workers on the master node and
  ! perform simple correctness test: check output result
  ! contains the correct number of exact matches.
  call check_output(Y, szy, szh, YY, szyy, mszyy, szhh, mrank, msize, master)

  deallocate(HH, XX, YY)
  
  if (mrank .eq. master) then
    write(*,*) 'control H: ', sum(H), minval(H), maxval(H)
    write(*,*) 'control X: ', sum(X), minval(X), maxval(X)
    write(*,*) 'control Y: ', sum(Y), minval(Y), maxval(Y)
    deallocate(H, X, Y)
    deallocate(mszxx, mszyy)
  endif

  call MPI_Finalize(ierr)
  if (ierr .ne. MPI_SUCCESS) then
    write(*,*) 'Error in MPI_Finalize'
    stop
  endif

end program main

