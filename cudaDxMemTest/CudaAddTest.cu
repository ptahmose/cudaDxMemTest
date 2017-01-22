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

CCudaAddTest::TestResult CCudaAddTest::AddTest_ManagedMemory2()
{
	return this->AddTest(&CCudaAddTest::AddTest_ManagedMemory2, "ManagedMemory2");
}

CCudaAddTest::TestResult CCudaAddTest::AddTest(void(CCudaAddTest::*testMethod)(TestResultBuilder&), const char* testMethodName)
{
	TestResultBuilder result;
	std::stringstream ss;
	ss << "*** Start of \"" << testMethodName << "\" ***";
	result.AddMessage(ss.str());

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

	ss = std::stringstream();
	ss << "*** End of \"" << testMethodName << "\" ***";
	result.AddMessage(ss.str());

	return result;
}

void CCudaAddTest::AddTest_DeviceMemory(TestResultBuilder& result)
{
	std::unique_ptr<int, free_delete> c((int*)malloc(this->numberOfElements * sizeof(int)));

	std::stringstream ss;
	auto timeIt = CTimeIt::CreateStarted();
	std::unique_ptr<int, free_cudaFree> dev_a((int*)CCudaUtils::cudaMalloc(this->numberOfElements * sizeof(int)));
	timeIt.Stop();
	ss << "cudaMalloc for " << this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	ss = std::stringstream();
	timeIt.Start();
	std::unique_ptr<int, free_cudaFree> dev_b((int*)CCudaUtils::cudaMalloc(this->numberOfElements * sizeof(int)));
	timeIt.Stop();
	ss << "cudaMalloc for " << this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	ss = std::stringstream();
	timeIt.Start();
	std::unique_ptr<int, free_cudaFree> dev_c((int*)CCudaUtils::cudaMalloc(this->numberOfElements * sizeof(int)));
	timeIt.Stop();
	ss << "cudaMalloc for " << this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	ss = std::stringstream();
	timeIt.Start();
	CCudaUtils::ThrowIfError(cudaMemcpy(dev_a.get(), this->a.get(), this->numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
	CCudaUtils::ThrowIfError(cudaMemcpy(dev_b.get(), this->b.get(), this->numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
	timeIt.Stop();
	ss << "cudaMemcpy host-to-device for " << 2 * this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	timeIt.Start();
	RunKernel(this->numberOfElements, dev_a.get(), dev_b.get(), dev_c.get());
	timeIt.Stop();
	AddTimeForOperation(result, timeIt, "running kernel");

	ss = std::stringstream();
	timeIt.Start();
	CCudaUtils::ThrowIfError(cudaMemcpy(c.get(), dev_c.get(), this->numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
	timeIt.Stop();
	ss << "cudaMemcpy device-to-host for " << this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	timeIt.Start();
	bool b = this->CheckResult(c.get());
	timeIt.Stop();
	AddTimeForOperation(result, timeIt, "checking result");
	result.SetSuccessful(b);
	AddResultCorrect(result, b);
}

void CCudaAddTest::AddTest_HostMemory(TestResultBuilder& result)
{
	std::unique_ptr<int, free_delete> c((int*)malloc(this->numberOfElements * sizeof(int)));

	auto timeIt = CTimeIt::CreateStarted();
	std::unique_ptr<void, free_hostunregister> registered_a(CCudaUtils::cudaHostRegister(this->a.get(), this->numberOfElements * sizeof(int), cudaHostRegisterDefault));
	std::unique_ptr<void, free_hostunregister> registered_b(CCudaUtils::cudaHostRegister(this->b.get(), this->numberOfElements * sizeof(int), cudaHostRegisterDefault));
	std::unique_ptr<void, free_hostunregister> registered_c(CCudaUtils::cudaHostRegister(c.get(), this->numberOfElements * sizeof(int), cudaHostRegisterDefault));

	int* dev_a, *dev_b, *dev_c;
	CCudaUtils::ThrowIfError(cudaHostGetDevicePointer(&dev_a, registered_a.get(), 0));
	CCudaUtils::ThrowIfError(cudaHostGetDevicePointer(&dev_b, registered_b.get(), 0));
	CCudaUtils::ThrowIfError(cudaHostGetDevicePointer(&dev_c, registered_c.get(), 0));
	timeIt.Stop();
	AddTimeForOperation(result, timeIt, "registering memory and get device-pointers");

	timeIt.Start();
	RunKernel(this->numberOfElements, dev_a, dev_b, dev_c);
	timeIt.Stop();
	AddTimeForOperation(result, timeIt, "running kernel");

	timeIt.Start();
	bool b = this->CheckResult(c.get());
	timeIt.Stop();
	AddTimeForOperation(result, timeIt, "checking result");

	result.SetSuccessful(b);
	AddResultCorrect(result, b);
}

void CCudaAddTest::AddTest_ManagedMemory(TestResultBuilder& result)
{
	std::stringstream ss;
	auto timeIt = CTimeIt::CreateStarted();
	std::unique_ptr<int, free_cudaFree> dev_a((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));
	timeIt.Stop();
	ss << "cudaMallocManaged for " << this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	ss = std::stringstream();
	timeIt.Start();
	std::unique_ptr<int, free_cudaFree> dev_b((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));
	timeIt.Stop();
	ss << "cudaMallocManaged for " << this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	ss = std::stringstream();
	timeIt.Start();
	std::unique_ptr<int, free_cudaFree> dev_c((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));
	timeIt.Stop();
	ss << "cudaMallocManaged for " << this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	ss = std::stringstream();
	timeIt.Start();
	CCudaUtils::ThrowIfError(cudaMemcpy(dev_a.get(), this->a.get(), this->numberOfElements * sizeof(int), cudaMemcpyDefault));
	CCudaUtils::ThrowIfError(cudaMemcpy(dev_b.get(), this->b.get(), this->numberOfElements * sizeof(int), cudaMemcpyDefault));
	timeIt.Stop();
	ss << "cudaMemcpy host-to-managed_memory for " << 2 * this->numberOfElements * sizeof(int) << " bytes";
	AddTimeForOperation(result, timeIt, ss.str());

	timeIt.Start();
	RunKernel(this->numberOfElements, dev_a.get(), dev_b.get(), dev_c.get());
	timeIt.Stop();
	AddTimeForOperation(result, timeIt, "running kernel");

	timeIt.Start();
	bool b = this->CheckResult(dev_c.get());
	timeIt.Stop();
	AddTimeForOperation(result, timeIt, "checking result");

	result.SetSuccessful(b);
	AddResultCorrect(result, b);
}

void CCudaAddTest::AddTest_ManagedMemory2(TestResultBuilder& result)
{
	std::unique_ptr<int, free_cudaFree> dev_a((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));
	CCudaUtils::ThrowIfError(cudaMemcpy(dev_a.get(), this->a.get(), this->numberOfElements * sizeof(int), cudaMemcpyDefault));
	CCudaUtils::ThrowIfError(cudaMemAdvise(dev_a.get(), this->numberOfElements * sizeof(int), cudaMemAdviseSetReadMostly, 0));
	//CCudaUtils::ThrowIfError(cudaMemPrefetchAsync(dev_a.get(), this->numberOfElements * sizeof(int), 0));

	std::unique_ptr<int, free_cudaFree> dev_b((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));
	CCudaUtils::ThrowIfError(cudaMemcpy(dev_b.get(), this->b.get(), this->numberOfElements * sizeof(int), cudaMemcpyDefault));
	CCudaUtils::ThrowIfError(cudaMemAdvise(dev_b.get(), this->numberOfElements * sizeof(int), cudaMemAdviseSetReadMostly, 0));
	//CCudaUtils::ThrowIfError(cudaMemPrefetchAsync(dev_b.get(), this->numberOfElements * sizeof(int), 0));

	std::unique_ptr<int, free_cudaFree> dev_c((int*)CCudaUtils::cudaMallocManaged(this->numberOfElements * sizeof(int)));

	RunKernel(this->numberOfElements, dev_a.get(), dev_b.get(), dev_c.get());

	bool b = this->CheckResult(dev_c.get());
	result.SetSuccessful(b);
	AddResultCorrect(result, b);
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

/*static*/void CCudaAddTest::AddResultCorrect(TestResultBuilder& result, bool successful)
{
	std::stringstream ss;
	ss << "verification of result : *** " << (successful ? "correct" : "wrong") << " ***";
	result.AddMessage(ss.str());
}

/*static*/void CCudaAddTest::AddTimeForOperation(TestResultBuilder& result, const CTimeIt& time_it, const std::string& operationName)
{
	std::stringstream ss;
	ss << "The operation \"" << operationName << "\" took " << time_it.GetElapsedTime_Microseconds() << " micro-seconds.";
	result.AddMessage(ss.str());
}