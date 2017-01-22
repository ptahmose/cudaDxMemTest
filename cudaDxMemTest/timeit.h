#pragma once

#include <Windows.h>
#include <cinttypes>

class CTimeIt
{
private:
	LARGE_INTEGER startTick;
	LARGE_INTEGER stopTick;
public:
	CTimeIt()
	{
		this->startTick.QuadPart = this->stopTick.QuadPart = 0;
	}

	static CTimeIt CreateStarted()
	{
		CTimeIt timeit;
		timeit.Start();
		return timeit;
	}

	void Start()
	{
		QueryPerformanceCounter(&this->startTick);
	}

	void Stop()
	{
		QueryPerformanceCounter(&this->stopTick);
	}

	std::int64_t GetElapsedTime_Microseconds() const
	{
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		LARGE_INTEGER elapsedMicroseconds;
		elapsedMicroseconds.QuadPart = this->stopTick.QuadPart - this->startTick.QuadPart;
		elapsedMicroseconds.QuadPart *= 1000000;
		return elapsedMicroseconds.QuadPart / frequency.QuadPart;
	}
};
