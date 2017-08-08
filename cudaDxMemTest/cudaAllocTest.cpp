#include "cudaAllocTest.h"
#include <cuda_runtime.h>
#include <vector>

CCudaAllocTest::CCudaAllocTest(int allocSize)
	: allocSize(allocSize)
{
}

void CCudaAllocTest::DoTest()
{
	std::vector<void*> devPointers;

	for (int i = 0;; ++i)
	{
		void* ptr;
		cudaError_t r = cudaMalloc(&ptr, this->allocSize);
		
	}
}