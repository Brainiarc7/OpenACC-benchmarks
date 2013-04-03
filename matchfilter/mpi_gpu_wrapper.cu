#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace std;

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		cout << "Usage: " << argv[0] << " <program>" << endl;
		return 0;
	}

	// Check if zero rank is dedicated to data collection only.
	int cpu_only_master = 0;
	char* cmaster = getenv("CPU_ONLY_MASTER");
	if (cmaster)
		cpu_only_master = atoi(cmaster);
	if (cpu_only_master)
		cpu_only_master = 1;

	// Get the number of available GPUs.
	int count = 0;
	cudaError_t err = cudaGetDeviceCount(&count);
	if (err != cudaSuccess)
	{
		cerr << "Error in cudaGetDeviceCount: " <<
			cudaGetErrorString(err) << endl;
		return 1;
	}

	// Get the MPI world size.
	char* csize = getenv("OMPI_COMM_WORLD_SIZE");
	if (!csize)
	{
		cerr << "Cannot determine the MPI world size. Are you using OpenMPI?" << endl;
		return 1;
	}

	// Check MPI world size does not exceed the
	// number of available GPUs.
	int size = atoi(csize);
	if (size - cpu_only_master > count)
	{
		cerr << "MPI world size exceeds the number of available GPUs" << endl;
		return 1;
	}

	// Get the MPI process rank.
	char* crank = getenv("OMPI_COMM_WORLD_RANK");
	if (!crank)
	{
		cerr << "Cannot determine the MPI process rank. Are you using OpenMPI?" << endl;
		return 1;
	}

	// Reset device to delete the currenly used CUDA context.
	err = cudaDeviceReset();
	if (err != cudaSuccess)
	{
		cerr << "Error in cudaDeviceReset: " << cudaGetErrorString(err) << endl;
		return 1;
	}

	// In CPU_ONLY_MASTER mode - switch the master node to
	// the CPU runmode.
	int rank = atoi(crank);
	if (cpu_only_master)
	{
		if (rank == 0)
		{
			const char* zero = "0";
			setenv("kernelgen_runmode", zero, 1);
		}
	}

	// Execute entire MPI process with the only one GPU visible,
	// which index is either the same as the MPI process rank or
	// less by 1, depending on the master node mode.
	// XXX: Note this mapping does not account the case of
	// multi-head cluster, where each node has several GPUs.
	// In this case one needs to mod rank by the number of available
	// GPUs.
	if (cpu_only_master && (rank > 0))
	{
		rank--;
		stringstream strrank;
		strrank << rank;
		string srank = strrank.str();
		setenv("CUDA_VISIBLE_DEVICES", srank.c_str(), 1);
	}
	else
		setenv("CUDA_VISIBLE_DEVICES", crank, 1);
	execv(argv[1], argv + 1); 

	return 0;
}

