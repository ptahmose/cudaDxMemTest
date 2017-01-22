#pragma once

#include <vector>
#include <cuda_runtime_api.h>
#include <memory>
#include "timeit.h"

class CCudaAddTest
{
public:
	class TestResult
	{
	protected:
		std::vector<std::string> messages;
		bool successful;

		TestResult() : successful(false) {}
	public:
		bool GetCompletedSuccessful() const { return this->successful; }
		const std::vector<std::string>& GetMessages() const { return this->messages; }
	};
private:
	class TestResultBuilder : public TestResult
	{
	public:
		void AddMessage(const std::string& str) { this->messages.push_back(str); }
		void SetSuccessful(bool b) { this->successful = b; }
	};

	struct free_delete
	{
		void operator()(void* x) { free(x); }
	};

	struct free_cudaFree
	{
		void operator()(void* x) { cudaFree(x); }
	};

	struct free_hostunregister
	{
		void operator()(void* x) { cudaHostUnregister(x); }
	};

	size_t numberOfElements;
	std::unique_ptr<int, free_delete> a;
	std::unique_ptr<int, free_delete> b;

	/*std::vector<int>	a;
	std::vector<int>	b;*/
public:
	CCudaAddTest(size_t numberOfInts);

	TestResult AddTest_DeviceMemory();
	TestResult AddTest_HostMemory();
	TestResult AddTest_ManagedMemory();
private:
	TestResult AddTest(void(CCudaAddTest::*testMethod)(TestResultBuilder&), const char* testMethodName);

	void AddTest_DeviceMemory(TestResultBuilder& result);
	void AddTest_HostMemory(TestResultBuilder& result);
	void AddTest_ManagedMemory(TestResultBuilder& result);

	void RunKernel(size_t numberOfElements, const int* devA, const int* devB, int* devC);

	static void AddTotalTime(TestResultBuilder& result, const char* testMethodName, const CTimeIt& time_it);

	bool CheckResult(const int* ptr);
};

