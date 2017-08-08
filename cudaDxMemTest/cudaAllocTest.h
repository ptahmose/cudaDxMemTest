#pragma once

class CCudaAllocTest
{
private:
	int allocSize;
public:
	CCudaAllocTest(int allocSize);

	void DoTest();
};