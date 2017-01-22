#pragma once


enum class CudaAddTest
{
	DeviceMemory = 1,
	HostMemory = 2,
	ManagedMemory=4
};

class COptions
{
public:
	COptions();

	bool ParseCommandLine(int argc, char** argv);

	bool IsCudaTestEnabled(CudaAddTest test) const;

	size_t GetSizeAllocAndFreeD3dTextures() const;
	size_t GetSizeAllocD3dTextures() const;
private:

};
