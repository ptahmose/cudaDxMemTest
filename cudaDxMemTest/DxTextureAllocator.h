#pragma once

#include <D3D11.h>
#include <vector>

class CDxTextureAllocator
{
private:
	ID3D11Device* pDevice;
	std::vector<ID3D11Texture2D*> textures;
public:
	CDxTextureAllocator();
	~CDxTextureAllocator();

	void Initialize();
	void AllocateAndFree(int size, int count);
	void AllocateAndKeep(int size, int count);
	void FreeHeldTextures();
};
