#include "options.h"
#include "optionparser.h"
#include <vector>
#include <iostream>
#include <regex>

struct Arg : public option::Arg
{
	static void printError(const char* msg1, const option::Option& opt, const char* msg2)
	{
		fprintf(stderr, "%s", msg1);
		fwrite(opt.name, opt.namelen, 1, stderr);
		fprintf(stderr, "%s", msg2);
	}

	static option::ArgStatus Required(const option::Option& option, bool msg)
	{
		if (option.arg != nullptr)
			return option::ARG_OK;

		if (msg) printError("Option '", option, "' requires an argument\n");
		return option::ARG_ILLEGAL;
	}
};

COptions::COptions()
{
}

bool COptions::ParseCommandLine(int argc, char** argv)
{
	enum  optionIndex { UNKNOWN, HELP, D3DALLOCTYPE, D3DALLOCSIZE, CUDAADDTESTS, CUDAADDTESTSIZE };

	static const option::Descriptor usage[] =
	{
		{ UNKNOWN, 0,"" , ""    ,option::Arg::None, "USAGE: cudaDxMemTest [options]\n\n"		"Options:" },
		{ HELP,    0,"" , "help",option::Arg::None, "  --help  \tPrint usage and exit." },
		{ D3DALLOCTYPE, 0, "a","d3d11alloctype",Arg::Required, "  --d3d11alloctype, -a  \tD3D11-allocation type, possible values are 'AllocateAndFree' and 'Allocate'." },
		{ D3DALLOCSIZE,    0,"d", "d3d11allocsize",Arg::Required, "  --d3d11allocsize, -d  \tThe size (in bytes) to be allocated by D3D11, you may use k, m or g as suffix." },
		{ CUDAADDTESTS,    0,"c", "cudaaddtests",Arg::Required, "  --cudaaddtests, -c  \tThe CUDA-Add-test to run." },
		{ CUDAADDTESTSIZE, 0,"s","cudasize" ,Arg::Required, "  --cudasize, -s  \tThe size (in bytes) to use in the CUDA-Add-test." },
		{ UNKNOWN, 0,"" ,  ""   ,option::Arg::None, "\nExamples:\n"
		"  example --unknown -- --this_is_no_option\n"
		"  example -unk --plus -ppp file1 file2\n" },
		{ 0,0,0,0,0,0 }
	};

	option::Stats stats(usage, argc, argv);

	std::vector<option::Option> options(stats.options_max);
	std::vector<option::Option> buffer(stats.buffer_max);

	option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);

	if (parse.error())
		return false;

	if (options[HELP] || argc == 0)
	{
		option::printUsage(std::cout, usage);
		return false;
	}

	bool b;
	for (int i = 0; i < parse.optionsCount(); ++i)
	{
		option::Option& opt = buffer[i];
		switch (opt.index())
		{
		case D3DALLOCSIZE:
			b = ParseAllocSize(opt.arg, this->d3dAllocSize);
			if (b == false)
			{
			}
			break;
		case D3DALLOCTYPE:
			b = ParseD3DAllocationType(opt.arg, this->d3dAllocationType);
			if (b == false)
			{
			}
			break;
		case CUDAADDTESTS:
			b = ParseCudaTestType(opt.arg, this->cudaAddTestType);
			if (b == false)
			{
			}
			break;
		}
	}

	return true;
}

bool COptions::IsCudaTestEnabled(CudaAddTestType type) const
{
	if (((int)type & this->cudaAddTestType) == (int)type)
	{
		return true;
	}

	return false;
}

bool COptions::IsD3DAllocationEnabled(D3DAllocationType type) const
{
	if (((int)type & this->d3dAllocationType) == (int)type)
	{
		return true;
	}

	return false;
}

size_t COptions::GetSizeAllocD3dTextures() const
{
	return this->d3dAllocSize;
}

bool COptions::ParseAllocSize(const std::string& str, size_t& size)
{
	std::regex argsRegex(R"(^\s*([0-9]*\.?[0-9]*)\s*([kKmMgG]?)\s*$)");
	std::smatch pieces_match;
	bool successful = false;
	if (std::regex_match(str, pieces_match, argsRegex))
	{
		if (pieces_match.size() == 3)
		{
			std::string s1 = pieces_match[1];
			std::string s2 = pieces_match[2];

			double val = std::stod(s1);
			char c = '\0';
			if (s2.length() > 0)
			{
				c = s2[0];
			}

			size = MakeSize(val, c);
			return true;
		}
	}

	return false;
}

/*static*/size_t COptions::MakeSize(double val, char c)
{
	int factor = 1;
	switch (tolower(c))
	{
	case 'k':
		factor = 1024;
		break;
	case 'm':
		factor = 1024 * 1024;
		break;
	case 'g':
		factor = 1024 * 1024 * 1024;
		break;
	}

	double d = val*factor;
	size_t s = (size_t)(d + .5);
	return s;
}

/*static*/bool COptions::ParseD3DAllocationType(const std::string& str, int& type)
{
	auto tokens = split(str, ",;|");

	type = 0;

	for (const std::string& token : tokens)
	{
		if (_strcmpi(token.c_str(), "AllocateAndFree") == 0)
		{
			type |= (int)D3DAllocationType::AllocateAndFree;
		}
		else if (_strcmpi(token.c_str(), "Allocate") == 0)
		{
			type |= (int)D3DAllocationType::Allocate;
		}
		else if (_strcmpi(token.c_str(), "All") == 0)
		{
			type |= (int)D3DAllocationType::AllocateAndFree | (int)D3DAllocationType::Allocate;
		}
		else
			return false;
	}

	return true;
}

/*static*/bool COptions::ParseCudaTestType(const std::string& str, int& type)
{
	auto tokens = split(str, ",;|");

	type = 0;

	for (const std::string& token : tokens)
	{
		if (_strcmpi(token.c_str(), "DeviceMemory") == 0)
		{
			type |= (int)CudaAddTestType::DeviceMemory;
		}
		else if (_strcmpi(token.c_str(), "HostMemory") == 0)
		{
			type |= (int)CudaAddTestType::HostMemory;
		}
		else if (_strcmpi(token.c_str(), "ManagedMemory") == 0)
		{
			type |= (int)CudaAddTestType::ManagedMemory;
		}
		else if (_strcmpi(token.c_str(), "All") == 0)
		{
			type |= (int)CudaAddTestType::DeviceMemory | (int)CudaAddTestType::HostMemory | (int)CudaAddTestType::ManagedMemory;
		}
		else
			return false;
	}

	return true;
}

std::vector<std::string> COptions::split(const std::string& str, const std::string& delims)
{
	std::vector<std::string> tokens;
	size_t prev = 0, pos = 0;
	do
	{
		pos = str.find_first_of(delims, prev);
		if (pos == std::string::npos) pos = str.length();
		std::string token = str.substr(prev, pos - prev);
		if (!token.empty())
		{
			tokens.push_back(trim(token));
		}

		prev = pos + 1;
	} while (pos < str.length() && prev < str.length());

	return tokens;
}

/*static*/std::string COptions::trim(const std::string &s)
{
	std::string::const_iterator it = s.begin();
	while (it != s.end() && isspace(*it))
		++it;

	std::string::const_reverse_iterator rit = s.rbegin();
	while (rit.base() != it && isspace(*rit))
		++rit;

	return std::string(it, rit.base());
}