
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "DxTextureAllocator.h"
#include <memory>
#include <algorithm>
#include "CudaAddTest.h"
#include "options.h"

/*

I get the following output:

Before D3D11-allocations
Total: 4294967296 bytes;  Free: 3554905292 bytes;  Used: 740062004
After D3D11-allocations-with-immediately free
Total: 4294967296 bytes;  Free: 2636463308 bytes;  Used: 1658503988			<-- after allocating D3D11-textures of around 2.5GB and immediately free'ing them
After D3D11-allocations
Total: 4294967296 bytes;  Free: 1823816908 bytes;  Used: 2471150388			<-- after allocating (and keeping) around 2.5GB of textures
cudaMalloc of 1073741824 bytes succeeded.
Total: 4294967296 bytes;  Free: 2454011084 bytes;  Used: 1840956212			<-- after allocating 1GB by cudaMalloc (NOTE: we have more free RAM than before...)
cudaMalloc of 1073741824 bytes succeeded.
Total: 4294967296 bytes;  Free: 1380269260 bytes;  Used: 2914698036
cudaMalloc of 1073741824 bytes succeeded.
Total: 4294967296 bytes;  Free: 265294642 bytes;  Used: 4029672654

 */


__global__ void addKernel2(int numberOfElements, int elementsPerInvocation, int *c, const int *a, const int *b)
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

static void CudaAddTest(int numberOfInts);
static void CudaAddTestUnifiedAddressing(int numberOfInts);

static void PrintMemInfo();

static void PrintResult(const CCudaAddTest::TestResult& result)
{
	for (const std::string& msg : result.GetMessages())
	{
		fputs(msg.c_str(), stdout);
		fputs("\n", stdout);
	}

	fputs("\n", stdout);
}
#if 0
int main(int argc,char** argv)
{
	COptions options;
	options.ParseCommandLine(argc-1, argv+1);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	PrintMemInfo();

	CCudaAddTest addTest(16*1024 * 1024);
	auto result = addTest.AddTest_DeviceMemory();
	PrintResult(result);

	PrintMemInfo();

	result = addTest.AddTest_HostMemory();
	PrintResult(result);

	PrintMemInfo();

	result = addTest.AddTest_ManagedMemory();
	PrintResult(result);

	PrintMemInfo();
	/*result = addTest.AddTest_ManagedMemory2();
	PrintResult(result);*/


	fprintf(stdout, "Before D3D11-allocations\n");
	PrintMemInfo();
	CDxTextureAllocator texture_allocator;
	texture_allocator.Initialize();
	texture_allocator.AllocateAndFree(1024 * 1024 * 25, 100);
	fprintf(stdout, "After D3D11-allocations-with-immediately free\n");
	PrintMemInfo();

	texture_allocator.AllocateAndKeep(1024 * 1024 * 25, 100);
	fprintf(stdout, "After D3D11-allocations\n");
	PrintMemInfo();


	// We allocate three buffers (with the number of ints specified),
	//  so the approx. memory consumption is n * 4 * 3.
	//  In this case, we allocate 3 times 1 GB.
	CudaAddTest(256 * 1024 * 1024);

	CudaAddTestUnifiedAddressing(32 * 1024 * 1024);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	return 0;
}
#endif
struct free_delete
{
	void operator()(void* x) { free(x); }
};

static void FillVector(int* v, size_t numberOfElements);
static bool CheckResult(int *c, const int *a, const int *b, unsigned int size);
static cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
static cudaError_t addWithCudaUnifiedAddressing(int *c, const int *a, const int *b, unsigned int size);

static void CudaAddTest(int numberOfInts)
{
	unsigned int ArraySize = numberOfInts;// 1024 * 1024 * 256;
	std::unique_ptr<int, free_delete> a((int*)malloc(ArraySize * sizeof(int)));
	FillVector(a.get(), ArraySize);
	std::unique_ptr<int, free_delete> b((int*)malloc(ArraySize * sizeof(int)));
	FillVector(b.get(), ArraySize);
	std::unique_ptr<int, free_delete> c((int*)malloc(ArraySize * sizeof(int)));

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c.get(), a.get(), b.get(), ArraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!\n");
		return;
	}

	bool isCorrect = CheckResult(c.get(), a.get(), b.get(), ArraySize);
	if (isCorrect)
	{
		fprintf(stdout, "Result is correct!\n");
	}
	else
	{
		fprintf(stdout, "Result is NOT correct!\n");
	}
}

static void CudaAddTestUnifiedAddressing(int numberOfInts)
{
	unsigned int ArraySize = numberOfInts;// 1024 * 1024 * 256;
	std::unique_ptr<int, free_delete> a((int*)malloc(ArraySize * sizeof(int)));
	FillVector(a.get(), ArraySize);
	std::unique_ptr<int, free_delete> b((int*)malloc(ArraySize * sizeof(int)));
	FillVector(b.get(), ArraySize);
	std::unique_ptr<int, free_delete> c((int*)malloc(ArraySize * sizeof(int)));

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCudaUnifiedAddressing(c.get(), a.get(), b.get(), ArraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCudaUnifiedAddressing failed!\n");
		return;
	}

	bool isCorrect = CheckResult(c.get(), a.get(), b.get(), ArraySize);
	if (isCorrect)
	{
		fprintf(stdout, "Result is correct!\n");
	}
	else
	{
		fprintf(stdout, "Result is NOT correct!\n");
	}
}

static void CudaAddTestUnifiedAddressing2(int numberOfInts)
{
	unsigned int ArraySize = numberOfInts;// 1024 * 1024 * 256;
	std::unique_ptr<int, free_delete> a((int*)malloc(ArraySize * sizeof(int)));
	FillVector(a.get(), ArraySize);
	std::unique_ptr<int, free_delete> b((int*)malloc(ArraySize * sizeof(int)));
	FillVector(b.get(), ArraySize);
	std::unique_ptr<int, free_delete> c((int*)malloc(ArraySize * sizeof(int)));

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCudaUnifiedAddressing(c.get(), a.get(), b.get(), ArraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCudaUnifiedAddressing failed!\n");
		return;
	}

	bool isCorrect = CheckResult(c.get(), a.get(), b.get(), ArraySize);
	if (isCorrect)
	{
		fprintf(stdout, "Result is correct!\n");
	}
	else
	{
		fprintf(stdout, "Result is NOT correct!\n");
	}
}

bool CheckResult(int *c, const int *a, const int *b, unsigned int size)
{
	for (unsigned int i = 0; i < size; ++i)
	{
		if (c[i] != a[i] + b[i])
			return false;
	}

	return true;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	int blockSize;
	int n_blocks;
	int elementsPerThread;

	cudaError_t cudaStatus;


	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	fprintf(stdout, "cudaMalloc of %llu bytes succeeded.\n", size * sizeof(int));
	PrintMemInfo();


	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	fprintf(stdout, "cudaMalloc of %llu bytes succeeded.\n", size * sizeof(int));
	PrintMemInfo();


	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	fprintf(stdout, "cudaMalloc of %llu bytes succeeded.\n", size * sizeof(int));
	PrintMemInfo();

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/*fprintf(stdout, "Memory allocated, press any key!");
	std::cin.ignore();
	fprintf(stdout, "\n");*/

	blockSize = 512;
	n_blocks = (std::min)((int)(size / blockSize + (size%blockSize == 0 ? 0 : 1)), 1024);
	elementsPerThread = size / (blockSize*n_blocks);

	addKernel2 << <n_blocks, blockSize >> > (size, (std::max)(elementsPerThread, 1), dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

cudaError_t addWithCudaUnifiedAddressing(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	int blockSize;
	int n_blocks;
	int elementsPerThread;

	cudaError_t cudaStatus;

	cudaStatus = cudaHostRegister(c, size * sizeof(int), cudaHostRegisterDefault);
	cudaStatus = cudaHostRegister((int*)a, size * sizeof(int), cudaHostRegisterDefault);
	cudaStatus = cudaHostRegister((int*)b, size * sizeof(int), cudaHostRegisterDefault);

	cudaStatus = cudaHostGetDevicePointer(&dev_c, c, 0);
	cudaStatus = cudaHostGetDevicePointer(&dev_a, (int*)a, 0);
	cudaStatus = cudaHostGetDevicePointer(&dev_b, (int*)b, 0);

	/*cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}*/

	blockSize = 512;
	n_blocks = (std::min)((int)(size / blockSize + (size%blockSize == 0 ? 0 : 1)), 1024);
	elementsPerThread = size / (blockSize*n_blocks);

	addKernel2 << <n_blocks, blockSize >> > (size, (std::max)(elementsPerThread, 1), dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaPointerAttributes attributes;
	cudaPointerGetAttributes(&attributes, dev_c);
	if (attributes.memoryType == cudaMemoryTypeDevice)
	{
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

Error:
	cudaPointerGetAttributes(&attributes, dev_c);
	if (attributes.memoryType == cudaMemoryTypeDevice)
	{
		cudaFree(dev_c);
	}
	else
	{
		cudaHostUnregister(c);
	}

	cudaHostUnregister((int*)a);
	cudaHostUnregister((int*)b);

	return cudaStatus;
}


void FillVector(int* v, size_t numberOfElements)
{
	for (size_t i = 0; i < numberOfElements; ++i)
	{
		*(v + i) = (int)i;
	}
}

static void PrintMemInfo()
{
	size_t total, free;
	cudaMemGetInfo(&free, &total);
	fprintf(stdout, "Total: %llu bytes;  Free: %llu bytes;  Used: %llu\n\n", total, free, total - free);
}