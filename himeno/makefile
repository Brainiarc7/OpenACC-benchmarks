all: himeno

himeno: himeno.f90
	kernelgen-gfortran -O3 $< -o $@

clean:
	rm -rf *.mod himeno
