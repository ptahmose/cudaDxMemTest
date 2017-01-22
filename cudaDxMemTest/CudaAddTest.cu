#include "CudaAddTest.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <memory>
#include <algorithm>
#include "cudaHelper.h"
#include "timeit.h"
#include <sstream>

__global__ void addKernel(int numberOfElements, int elementsPerInvocation, int *c, const int *a, const int *b)
{
	for (int i = 0; i < elementsPerInvocation; ++i)
	{
		int idx = blockIdx.x * blockDim.x * elementsPerInvocation + threadIdx.x*elementsPerInvocation;
		if (idx < numberOfElements)
		{
			c[idx + i] = a[idx + i] + b[idx + i];
		}
	}
}

CCudaAddTest::CCudaAddTest(size_t numberOfInts)
	: numberOfElements(numberOfInts), a((int*)malloc(numberOfInts * sizeof(int))), b((int*)malloc(numberOfInts * sizeof(int)))
{
	int* p = this->a.get();
	for (size_t i = 0; i < numberOfInts; ++i)
	{
		p[i] = (int)i;
	}

	p = this->b.get();
	for (size_t i = 0; i < numberOfInts; ++i)
	{
		p[i] = (int)(i + 42);
	}
}

CCudaAddTest::TestResult CCudaAddTest::AddTest_DeviceMemory()
{
	return this->AddTest(&CCudaAddTest::AddTest_DeviceMemory, "DeviceMemory");
}

CCudaAddTest::TestResult CCudaAddTest::AddTest_HostMemory()
{
	return this->AddTest(&CCudaAddTest::AddTest_HostMemory, "HostMemory");
}

CCudaAddTest::TestResult CCudaAddTest::AddTest_ManagedMemory()
{
	return this->AddTest(&CCudaAddTest::AddTest_ManagedMemory, "ManagedMemory");
}

CCudaAddTest::TestResult CCudaAddTest::AddTest(void(CCudaAddTest::*testMethod)(TestResultBuilder&), const char* testMethodName)
{
	TestResultBuilder result;

	auto timeIt = CTimeIt::CreateStarted();
	try
	{
		(this->*testMethod)(result);
	}
	catch (cudaErrorException& excp)
	{
		std::stringstream ss;
		ss << "Test terminated with CUDA-exception: \"" << excp.what() << "\".";
		result.AddMessage(ss.str());
	}

	timeIt.Stop();

	AddTotalTime(result, testMethodName, timeIt);

	return result;
}

void CCudaAddTest::AddTest_DeviceMemory(TestResultBuilder& result)
{
	std::unique_ptr<int, free_delete> c((int*)malloc(this->numberOfElements * sizeof(int)));

	std::unique_ptr<int, free_cudaFree> dev_a((int*)CCudaUtils::cudaMalloc(this->numberOfElements * sizeof(int)));
	std::unique_ptr<int, free_cudaFree> dev_b((int*)CCudaUtils::cudaMalloc(this->numberOfElements * sizeof(int)));
	std::unique_ptr<int, free_cudaFree> dev_c((int*)CCudaUtils::cudaMalloc(this->numberOfElements * sizeof(int)));

	CCudaUtils::ThrowIfError(cudaMemcpy(dev_a.get(), this->a.get(), this->numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
	CCudaUtils::ThrowIfError(cudaMemcpy(dev_b.get(), this->b.get(), this->numberOfElements * sizeof(int), cudaMemcpyHostToDevice));

	RunKernel(this->numberOfElements, dev_a.get(), dev_b.get(), dev_c.get());

	CCudaUtils::ThrowIfError(cudaMemcpy(c.get(), dev_c.get(), this->numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));

	bool b = this->CheckResult(c.get());
	result.SetSuccessful(b);
}

void CCudaAddTest::AddTest_HostMemory(TestResultBuilder& result)
{
	std::unique_ptr<int, free_delete> c((int*)malloc(this->numberOfElements * sizeof(int)));

	std::unique_ptr<void, free_hostunregister> registered_a(CCudaUtils::cudaHostRegister(this->a.get(), this->numberOfElements * sizeof(int), cudaHostRegisterDefault));
	std::unique_ptr<void, free_hostunregister> registered_b(CCudaUtils::cudaHostRegister(this->b.get(), this->numberOfElements * sizeof(int), cudaHostRegisterDefault));
	std::unique_ptr<void, free_hostunregister> registered_c(CCudaUtils::cudaHostRegister(c.get(), this->numberOfElements * sizeof(int), cudaHostRegisterDefault));

	int* dev_a, *dev_b, *dev_c;
	CCudaUtils::ThrowIfError(cudaHostGetDevicePointer(&dev_a, registered_a.get(), 0));
	CCudaUtils::ThrowIfError(cudaHostGetDevicePointer(&dev_b, registered_b.get(), 0));
	CCudaUtils::ThrowIfError(cudaHostGetDevicePointer(&dev_c, registered_c.get(), 0));

	RunKernel(this->numberOfElements, dev_a, dev_b, dev_c);

	bool b = this->CheckResult(c.get());
	result.SetSuccessful(b);
}

void CCudaAddTest::AddTest_ManagedMemory(TestResultBuilder& result)
{
	std::unique_ptr<int, free_cudaFree> dev_a((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));
	std::unique_ptr<int, free_cudaFree> dev_b((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));
	std::unique_ptr<int, free_cudaFree> dev_c((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));

	CCudaUtils::ThrowIfError(cudaMemcpy(dev_a.get(), this->a.get(), this->numberOfElements * sizeof(int), cudaMemcpyDefault));
	CCudaUtils::ThrowIfError(cudaMemcpy(dev_b.get(), this->b.get(), this->numberOfElements * sizeof(int), cudaMemcpyDefault));

	RunKernel(this->numberOfElements, dev_a.get(), dev_b.get(), dev_c.get());

	bool b = this->CheckResult(dev_c.get());
	result.SetSuccessful(b);
}

void CCudaAddTest::RunKernel(size_t numberOfElements, const int* devA, const int* devB, int* devC)
{
	int blkSize = 512;
	int nBlocks = (std::min)((int)(numberOfElements / blkSize + (numberOfElements%blkSize == 0 ? 0 : 1)), 1024);
	int elementsPerThread = numberOfElements / (blkSize*nBlocks);
	addKernel << <nBlocks, blkSize >> > (numberOfElements, (std::max)(elementsPerThread, 1), devC, devA, devB);

	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	CCudaUtils::ThrowIfError(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	CCudaUtils::ThrowIfError(cudaStatus);
}

bool CCudaAddTest::CheckResult(const int* ptr)
{
	const int* p_a = this->a.get();
	const int* p_b = this->b.get();
	for (size_t i = 0; i < this->numberOfElements; ++i)
	{
		int c = *(ptr + i);
		int s = p_a[i] + p_b[i];
		if (c != s)
		{
			return false;
		}
	}

	return true;
}

/*static*/void CCudaAddTest::AddTotalTime(TestResultBuilder& result, const char* testMethodName, const CTimeIt& time_it)
{
	std::stringstream ss;
	ss << "Total elapsed time for method \"" << testMethodName << "\": " << time_it.GetElapsedTime_Microseconds() << " micro-seconds.";
	result.AddMessage(ss.str());
}