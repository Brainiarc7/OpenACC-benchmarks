IKIND = 4
RKIND = 8

all: matchfilter matchfilter_mpi

matchfilter: matchfilter.F90
	kernelgen-gfortran -DIKIND=$(IKIND) -DRKIND=$(RKIND) -O3 $< -o $@

matchfilter_mpi: matchfilter_mpi.F90 mpi_gpu_wrapper
	OMPI_FC=kernelgen-gfortran mpif90 -DIKIND=$(IKIND) -DRKIND=$(RKIND) -O3 $< -o $@

mpi_gpu_wrapper: mpi_gpu_wrapper.cu
	nvcc $< -o $@

clean:
	rm -rf matchfilter matchfilter_mpi mpi_gpu_wrapper

