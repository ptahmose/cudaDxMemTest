#include <stdio.h>
#include "DxTextureAllocator.h"
#include <memory>
#include <algorithm>
#include "CudaAddTest.h"
#include "options.h"

int main(int argc, char** argv)
{
	COptions options;
	options.ParseCommandLine(argc - 1, argv + 1);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	const int SIZE_TEXTURE = 1024 * 1024 * 10;

	CDxTextureAllocator texture_allocator;
	texture_allocator.Initialize();
	if (options.IsD3DAllocationEnabled(D3DAllocationType::AllocateAndFree))
	{
		texture_allocator.AllocateAndFree(SIZE_TEXTURE, (std::max)((int)options.GetSizeAllocD3dTextures() / SIZE_TEXTURE, 1));
	}


}