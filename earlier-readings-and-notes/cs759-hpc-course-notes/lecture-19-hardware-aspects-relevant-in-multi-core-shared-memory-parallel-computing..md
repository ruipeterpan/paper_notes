# Lecture 19: Hardware aspects relevant in multi-core, shared memory parallel computing.

## Lecture Summary

* Last time
  * GPU computing via thrust & CUB
* Today
  * Final project proposal discussion
  * Parallel computing on the CPU: Hardware & OpenMP generalities

## Multi-core Parallel Computing with OpenMP

![Opportunities for efficiency gains](<../../.gitbook/assets/Screen Shot 2021-03-22 at 11.56.06 AM.png>)

* OpenMP targets parallelism on SMP architectures
* It is handy when
  * You have a multi-core processor, say 16 cores/socket (go beyond that and we suffer from diminishing returns due to overheads)
  * Might have multiple sockets, say 2
  * You have a good amount of system memory, say 64 GB
* Processes and threads are similar in the sense that they are both independent sequences of execution
  * OpenMP touches on threads, while MPI touches on processes
  * Threads of the same process run in a shared memory space and they have one translation page. Processes, on the other hand, run in separate memory spaces.
* We want to use OpenMP for both data parallelism and task parallelism
  * Data parallelism: The processing of a large amount of data elements can be done in parallel
  * Task parallelism: The execution of a collection of tasks can be performed in parallel

![Hello world for OpenMP](<../../.gitbook/assets/Screen Shot 2021-03-22 at 12.09.44 PM.png>)

* The OMP parallel region is similar to a CUDA kernel: both are executed by threads
  * A major difference
    * Variables inside GPU kernel are truly local variables, stored in registers
    * OMP variables in a parallel region may or may not be visible to other threads executing the code of the parallel region: the scoping is tricky
* `#include <omp.h>`
* Most OpenMP constructs are compiler directives. In C/C++, they take the form of `pragmas`
* Programming model: A master thread spawns a team of threads

![](<../../.gitbook/assets/Screen Shot 2021-03-22 at 12.12.27 PM.png>)







