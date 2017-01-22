#include "options.h"
#include "optionparser.h"
#include <vector>
#include <iostream>

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
	enum  optionIndex { UNKNOWN, HELP, D3DALLOCSIZE, CUDAADDTESTS, CUDAADDTESTSIZE };

	static const option::Descriptor usage[] =
	{
		{ UNKNOWN, 0,"" , ""    ,option::Arg::None, "USAGE: example [options]\n\n"
		"Options:" },
		{ HELP,    0,"" , "help",option::Arg::None, "  --help  \tPrint usage and exit." },
		{ D3DALLOCSIZE,    0,"d", "d3d11allocsize",Arg::Required, "  --command, -c  \tMay either be 'fit' or 'generate'." },
		{ CUDAADDTESTS,    0,"c", "cudatests",Arg::Required, "  --input, -i  \tThe input file." },
		{ CUDAADDTESTSIZE, 0,"s","cudasize" ,Arg::Required, "  --svgoutput, -s  \tThe SVG-output filename." },
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
}

