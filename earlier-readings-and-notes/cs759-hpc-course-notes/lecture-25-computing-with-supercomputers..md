# Lecture 25: Computing with Supercomputers.

## Lecture Summary

* Last time
  * Critical thinking, and similar tricks for speeding up your code
* Today
  * Wrapping up critical thinking
    * Case study



## Critical Thinking

* Once again, a lot of interesting stuff in the [slides](https://uwmadison.app.box.com/s/vc7xn1juqed1nyi1a323t0ytma55wiyd). Here's a quick summary of what's covered
* Basic optimizations
* Exploiting Instruction-Level Parallelism (ILP)
  * Hazards: Structural, data dependency, control
* Pipelining
* Loop unrolling
* Loop unrolling with reassociation
* Loop unrolling with separate accumulators
* Vector instructions (use fat registers to perform the same operation on all variables stored in register)
* Branch prediction
* On an unrelated note: I took [CS 61C @ Berkeley](https://cs61c.org/su19/) in summer 2019, and one of the project assignments is [Performance Programming](https://cs61c.org/su19/projects/proj4/) and the optimization techniques used include:
  * Profiling & Amdahl's Law
  * Unrolling & Other Optimizations
  * SIMD Instructions
  * OpenMP
* I transferred the credits of CS 61C to meet the requirements for CS 354 at UW-Madison. Looking back from the top of the mountain, CS 61C really covered most of the stuff in CS 252, 354, a bit of 352, 552, and 537, and even things in 759. I'm not saying that our courses are bad (the classes I mentioned above, particularly the more advanced ones, spent more time going in-depth) but damn. Berkeley is so good. So good.

## MPI

![Nomenclatures](<../../.gitbook/assets/Screen Shot 2021-04-11 at 12.11.46 AM.png>)

![HPC vs. HTC](<../../.gitbook/assets/Screen Shot 2021-04-11 at 12.13.41 AM.png>)
