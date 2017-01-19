#pragma once

#include <D3D11.h>

class CDxTextureAllocator
{
private:
	ID3D11Device* pDevice;
public:
	CDxTextureAllocator();
	~CDxTextureAllocator();

	void Initialize();
	void AllocateAndFree(int size, int count);
};
