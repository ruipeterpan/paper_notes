# Lecture 24: Critical Thinking. Code Optimization Aspects.

## Lecture Summary

* Last time
  * Parallel computing, multi-core: how caches come into play
  * Critical thinking, and similar tricks for speeding up your code
* Today
  * Critical thinking, and other similar tricks for speeding up your code

## Know your hardware

![Know your bandwidth/latency](<../../.gitbook/assets/Screen Shot 2021-04-10 at 11.39.24 PM.png>)

## Choose the right algorithm

* When working on a problem, there's more to performance than asymptotic complexity
  * Because asymptotic complexity is most often defined by the number of operations
  * Memory transactions are rarely considered: they are specific to the hardware
* Assess the arithmetic intensity associated with your problem

![](<../../.gitbook/assets/Screen Shot 2021-04-10 at 11.43.21 PM.png>)

* Simple optimization: Fusing transformations and do not bring data into the cache twice

## Compiler

* Aggressive optimizations done by the compiler might change the behavior of your code
* To help the compiler:
  * Allow it to see as much code as possible
  * Provide flags to convey information (e.g., the target architecture)
* There are a lot of amazing things covered in this lecture. The takeaways are:
  * Compilers are fantastic
  * Know them better to use them wisely
* A quick example is down below. Refer to the [slides](https://uwmadison.app.box.com/s/kapdp4qt18c6869dnaenlo35iytnlp2r) for a lot more fun facts

![](<../../.gitbook/assets/Screen Shot 2021-04-10 at 11.59.06 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-04-10 at 11.59.17 PM.png>)
