// Package concurrency provides a small bounded-parallelism helper.
package concurrency

import "sync"

// Limiter runs jobs with bounded parallelism.
type Limiter interface {
	// Run blocks until a slot is free, then runs job in its own goroutine and
	// returns immediately. A job's error is not surfaced; callers that need a
	// result must hand it back themselves.
	Run(job func() error)
	// WaitForAll blocks until every job passed to Run has finished.
	WaitForAll()
}

type jobLimiter struct {
	slots chan struct{}
	wg    sync.WaitGroup
}

// NewJobLimiter returns a Limiter that runs at most limit jobs at once.
// Values below 1 are treated as 1.
func NewJobLimiter(limit int) Limiter {
	if limit < 1 {
		limit = 1
	}

	return &jobLimiter{slots: make(chan struct{}, limit)}
}

func (l *jobLimiter) Run(job func() error) {
	l.slots <- struct{}{}

	l.wg.Add(1)

	go func() {
		defer func() {
			<-l.slots
			l.wg.Done()
		}()

		_ = job()
	}()
}

func (l *jobLimiter) WaitForAll() {
	l.wg.Wait()
}
