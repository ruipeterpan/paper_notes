# Lecture 20: Multi-core Parallel Computing with OpenMP. Parallel Regions.

## Lecture Summary

* Last time: OpenMP generalities
* This time: OpenMP nuts & bolts

## OpenMP

![Compiler directives examples \(the directive goes behind \`\#pragma omp\`\)](../../.gitbook/assets/screen-shot-2021-03-22-at-12.24.37-pm.png)

![User-level run time routines](../../.gitbook/assets/screen-shot-2021-03-22-at-12.26.12-pm.png)

![Environment variables. This helps with bypassing the run-time function calls, but using env vars does not allow for dynamic OpenMP behavior. A function call overrides an env var setting, though.](../../.gitbook/assets/screen-shot-2021-03-22-at-12.32.59-pm.png)

* OpenMP: portable and scalable model for shared memory parallel applications
  * No need to dive deep and work with POSIX pthreads
  * Under the hood, the compiler translates OpenMPfunctions and directives to pthread calls
* Structured block and OpenMP construct are the two sides of the “parallel region” coin
* In a structured block, the only "branches" allowed are exit\(\) function calls. There is an implicit barrier after each structured block where threads wait on each other.

![](../../.gitbook/assets/screen-shot-2021-03-22-at-12.55.23-pm.png)

![](../../.gitbook/assets/screen-shot-2021-03-22-at-12.57.03-pm.png)

### Nested Parallelism

![](../../.gitbook/assets/screen-shot-2021-03-22-at-12.59.46-pm.png)

* The nested parallelism behavior can be controlled by using the OpenMP API
* The single directive identifies a section of the code that must be run by a single thread
  * The difference between single and master is that in single, the code is executed by whichever thread reaches the region first
  * Another diff is that for single, there is an implicit barrier upon completion of the region

![](../../.gitbook/assets/screen-shot-2021-03-22-at-1.06.02-pm.png)

### Work Sharing

* Work sharing is a general term used in OpenMP to describe the distribution of work across threads
* The three main constructs for automatic work division are:
  * omp for
  * omp sections
  * omp task

### omp for

![](../../.gitbook/assets/screen-shot-2021-03-22-at-1.12.29-pm.png)

* A \#pragma omp for inside a \#pragma omp parallel is equivalent to \#pragma omp parallel for
* Most OpenMP implementations use default block partitioning, where each thread is assigned roughly n/thread\_count iterations. This may lead to load imbalance if the work per iteration varies
  * The schedule clause comes to the rescue!
  * Usage example: \#pragma omp parallel for schedule\(static, 8\)

![](../../.gitbook/assets/screen-shot-2021-03-22-at-1.19.13-pm.png)

![Effects of different schedules, assuming 3 threads](../../.gitbook/assets/screen-shot-2021-03-22-at-1.20.49-pm.png)

![Choosing a schedule](../../.gitbook/assets/screen-shot-2021-03-22-at-1.21.28-pm.png)

* OpenMP will only parallelize for loops that are in canonical form. Counterintuitive behavior may happen
* The collapse clause supports collapsing the embedded loops into one uber loop
  * For example, if the outer loop has 10 iters, the inner loop has 10^7 iters, and we have 32 threads: parallelizing the outer loop is bad \(10&lt;32\), parallelizing the inner loop is good, but we can do better using collapse

![](../../.gitbook/assets/screen-shot-2021-03-22-at-1.29.16-pm.png)

### omp sections

![](../../.gitbook/assets/screen-shot-2021-03-22-at-1.30.42-pm.png)

![](../../.gitbook/assets/screen-shot-2021-03-22-at-1.31.20-pm.png)

![](../../.gitbook/assets/screen-shot-2021-03-22-at-1.33.09-pm.png)

![](../../.gitbook/assets/screen-shot-2021-03-22-at-1.33.20-pm.png)

