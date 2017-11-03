/*
 * Barrier.cpp -- simple cyclic barrier implementation
 *
 * borrowed from http://studenti.ing.unipi.it/~s470694/a-cyclic-thread-barrier/
 *
 * (c) 2017 Prof Dr Andreas Müller, Hochschule Rapperswil
 */
#include "Barrier.hpp"

Barrier::Barrier(int n) : n_threads(n), current(0) {
	counts[0] = 0;
	counts[1] = 0;
}

void	Barrier::await() {
	std::unique_lock<std::mutex>	lock(_mutex);
	int	my = current;
	counts[my]++;
	if (counts[my] < n_threads) {
		_condition.wait(lock);
	} else {
		// we are the last thread, so we switch to the other counter
		current ^= 1;
		counts[current] = 0;
		_condition.notify_all();
	}
}

Barrier::~Barrier() {
}
