#pragma once
#include <exception>
#include <cuda_runtime_api.h>

class cudaErrorException : public std::exception
{
private:
	cudaError_t cuda_error;
public:
	cudaErrorException(cudaError_t cuda_error) : cuda_error(cuda_error)
	{}

	const char* what() const throw() { return cudaGetErrorString(this->cuda_error); }
};

class CCudaUtils
{
public:
	static void* cudaMalloc(size_t size) throw(cudaErrorException&)
	{
		void* vp;
		cudaError_t cudaStatus = ::cudaMalloc(&vp, size);
		if (cudaStatus != cudaSuccess)
		{
			throw cudaErrorException(cudaStatus);
		}

		return vp;
	}

	static void* cudaHostRegister(void* ptr, size_t size, unsigned flags)
	{
		cudaError_t cudaStatus = ::cudaHostRegister(ptr, size, flags);
		if (cudaStatus != cudaSuccess)
		{
			throw cudaErrorException(cudaStatus);
		}

		return ptr;
	}

	static void* cudaMallocManaged(size_t size, unsigned flags = cudaMemAttachGlobal)
	{
		void* vp;
		cudaError_t cudaStatus = ::cudaMallocManaged(&vp, size, flags);
		if (cudaStatus != cudaSuccess)
		{
			throw cudaErrorException(cudaStatus);
		}

		return vp;
	}

	static void ThrowIfError(cudaError_t cudaStatus) throw(cudaErrorException&)
	{
		if (cudaStatus != cudaSuccess)
		{
			throw cudaErrorException(cudaStatus);
		}
	}
};