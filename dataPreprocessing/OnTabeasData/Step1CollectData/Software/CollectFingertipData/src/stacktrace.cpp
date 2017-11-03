/*
 * stacktrace.cpp -- handler for segmentation faults
 *
 * (c) 2014 Prof Dr Andreas Mueller, Hochschule Rapperswil
 */
#include <execinfo.h>
#include <cstdio>
#include <cstdlib>

extern "C" void	stacktrace(int sig) {
	if (sig > 0) {
		fprintf(stderr, "stacktrace caused by signal %d\n", sig);
	}
	void	*frames[50];
	int	size = backtrace(frames, sizeof(frames));
	char	**messages = backtrace_symbols(frames, size);
	if (NULL != messages) {
		for (int i = 0; i < size; i++) {
			fprintf(stderr, "[%d] %s\n", i, messages[i]);
		}
	} else {
		fprintf(stderr, "cannot obtain symbolic information");
	}
	if (sig > 0) {
		exit(EXIT_FAILURE);
	}
}

