#pragma once
#include <string>
#include <vector>


enum class CudaAddTestType :int
{
	None = 0,
	DeviceMemory = 1,
	HostMemory = 2,
	ManagedMemory = 4
};

enum class D3DAllocationType :int
{
	None = 0,
	AllocateAndFree = 1,
	Allocate = 2
};

class COptions
{
private:
	size_t d3dAllocSize;
	int d3dAllocationType;
	int cudaAddTestType;
public:
	COptions();

	bool ParseCommandLine(int argc, char** argv);

	bool IsCudaTestEnabled(CudaAddTestType) const;
	bool IsD3DAllocationEnabled(D3DAllocationType) const;

	size_t GetSizeAllocD3dTextures() const;

	bool DoCudaAllocTest() const;
private:
	static bool ParseAllocSize(const std::string& str, size_t& size);
	static bool ParseD3DAllocationType(const std::string& str, int& type);
	static bool ParseCudaTestType(const std::string& str, int& type);
	static size_t MakeSize(double val, char c);
	static std::vector<std::string> split(const std::string& str, const std::string& delims);
	static std::string trim(const std::string &s);
};
