#include "DxTextureAllocator.h"
#include <cmath>
#include <algorithm>

using namespace std;

CDxTextureAllocator::CDxTextureAllocator()
	: pDevice(nullptr)
{
}

CDxTextureAllocator::~CDxTextureAllocator()
{
	if (this->pDevice != nullptr)
	{
		this->pDevice->Release();
	}
}

void CDxTextureAllocator::Initialize()
{
	HRESULT hr = D3D11CreateDevice(
		nullptr,
		D3D_DRIVER_TYPE_HARDWARE,
		nullptr,
		D3D11_CREATE_DEVICE_BGRA_SUPPORT,
		nullptr,
		0,
		D3D11_SDK_VERSION,
		&this->pDevice,
		nullptr,
		nullptr);
}

void CDxTextureAllocator::AllocateAndFree(int size, int count)
{
	// we allocate a RGBX32-surface, calculate approx. width and height for the given size (in bytes)
	int pixelNumber = size / 4;
	int width = (std::max)((int)(0.5 + sqrt(pixelNumber)), 1);
	int height = width;

	D3D11_TEXTURE2D_DESC desc;
	memset(&desc, sizeof(desc), 0);
	desc.Width = width;
	desc.Height = height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	desc.Format = DXGI_FORMAT_B8G8R8X8_UNORM;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_IMMUTABLE;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;

	void* pBuffer = malloc(width * 4 * height);

	D3D11_SUBRESOURCE_DATA subResData;
	memset(&subResData, sizeof(subResData), 0);
	subResData.pSysMem = pBuffer;
	subResData.SysMemPitch = width * 4;

	ID3D11Texture2D* pD3dTexture = nullptr;

	for (int i = 0; i < count; ++i)
	{
		HRESULT hr = this->pDevice->CreateTexture2D(&desc, &subResData, &pD3dTexture);
		pD3dTexture->Release();
	}
}
