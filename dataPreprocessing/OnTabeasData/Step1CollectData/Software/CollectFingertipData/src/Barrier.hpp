/*
 * Barrier.h -- Barrier class
 *
 * (c) 2017 Prof Dr Andreas Mueller, Hochschule Rapperswil
 */
#ifndef _Barrier_h
#define _Barrier_h

#include <mutex>
#include <condition_variable>

/**
 * \brief a Barrier implementation
 */
class Barrier {
	std::mutex	_mutex;
	std::condition_variable	_condition;
	const int	n_threads;
	int 	counts[2];
	int	current;

	Barrier(const Barrier&);
	Barrier&	operator=(const Barrier&);
public:
	Barrier(int n);
	~Barrier();
	void	await();
};

#endif /* _Barrier_h */
