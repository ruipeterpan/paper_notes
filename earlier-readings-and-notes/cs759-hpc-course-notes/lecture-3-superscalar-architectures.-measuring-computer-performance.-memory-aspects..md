# Lecture 3: Superscalar architectures. Measuring Computer Performance. Memory Aspects.

## Lecture Summary

* Registers
* ILP (Instruction Level Parallelism), focus on pipelining, also mentions OOOE and multiple-issue&#x20;
* TLP (Thread Level Parallelism), and HTT (Intel Hyper-Threading Technology) discussion
* Execution times

## Registers

* A register is a hardware asset whose role is to store information (data value/instruction)
* It's the storage type with the shortest latency (closest to CU & ALU)
* The number & size of registers used are specific to an ISA
* Take-home message: When processing one instruction, the chip needs all instruction-related operands in registers

Types of registers include:

* Instruction Register (IR): Holds the instruction that is executed
* Program Counter (PC): Holds the address of the instruction executed next
* Memory Data Register (MDR): Holds data read in from memory/produced by the ALU and waiting to be stored in memory
* Memory Address Register (MAR): Holds the address of RAM memory location where I/O data is supposed to be read in/written out
* Return Address (RA): The address where upon finishing a sequence of instructions, the execution should jump and commence with the execution of subsequent instruction
* Others include registers for:
  * Subroutine Arguments
  * Temporary Variables
  * Saved Temporary Variables
* Several other registers for handling function calls are:
  * Stack Pointer (SP): Holds an address to the top of the stack
  * Global Pointer (GP): Holds on to a global pointer that points into the middle of a 64KB block of memory in the heap that holds constants and global variables
  * Frame Pointer (FP): Holds an address that points to the beginning of the procedure frame (e.g., the previous SP before this function changed its value)

## ILP: Pipelining

* The concept of the "clock cycle": In a factory assembly line, it's the time from the moment a station takes an input to the moment the output leaves the station. In a processor, the clock cycle is the time between two ticks of the internal clock of the microprocessor/chip. The clock speed is typically measured in Hz (pulses/s).&#x20;
* The FDX cycle can be expanded to a five-stage process:
  * Fetch instruction
  * Decode instruction
  * Data access
  * Execute the operation
  * Write-back into register file
* Pipelining idea: Different stages of different instructions can be worked upon simultaneously

Consider these instructions:

```
sw $t0,  0($s2) //store what is in register $t0 at mem location  0 bytes from address in register $s2
sw $t1, 32($s2) //store what is in register $t1 at mem location 32 bytes from address in register $s2
sw $t2, 64($s2) //store what is in register $t0 at mem location 64 bytes from address in register $s2
```

![Case 1: No pipelining](<../../.gitbook/assets/Screen Shot 2021-01-29 at 4.03.39 PM.png>)

![Case 2: With pipelining](<../../.gitbook/assets/Screen Shot 2021-01-29 at 4.04.04 PM.png>)

Ideally, in a balanced pipeline, each component/stage takes the same amount of time for completion to prevent a "bottleneck". Today, a typical pipeline depth is \~12-15 stages. Using pipelining gives a speed up (duh), and it also requires no changes on the user-code level.

Things can go south, though...

### Structural Hazards

An example is resource contention (e.g., two pipelined instructions have stages that need to use the same special register at the same time). The solutions are:

1. Commandeer a register for temporary use: Fortuitous
2. Serialize the access (introduce a bubble in the pipeline): Guaranteed to work, but introduces slowdown
3. OOOE (Out of Order Execution) performed statically at compile time or dynamically at run time: Good compromise

### Data Hazards

```
add  $t0, $t2, $t4   // $t0 = $t2 + $t4
addi $t3, $t0, 16    // $t3 = $t0 + 16 “add immediate instruction”
```

In the example above, you might think that `t0` is unavailable until the first instruction completely finishes. This is partially true: actually, `t0` becomes available after stage 3 of the pipeline (after it goes through the ALU). A solution, Intermediate Result Forwarding, makes the result in the ALU available to other stages of the pipeline right away. This is not a panacea, and occasionally we still need to do bubbling for a couple of cycles. OOOE also definitely helps here.

### Control Hazards

For instance, if there's an if statement in the C code (`if (sin(x)/x > 0.5`), we don't know the next instruction until the computation completes, which takes a few cycles. Bubbling again works, but it introduces slowdown. An alternative is to do branch prediction. There are two versions:

1. Static Branch Prediction: Always predict that the branch will not be taken and schedule accordingly (always the then branch, never the else branch). In some other cases (e.g., a do-while loop), it makes more sense.
2. Dynamic Branch Prediction: Make the branching decision based on recent history. In some cases, the accuracy rate can reach 90%.

## ILP: Multiple-Issue

```
int a, b;
float c, d;
//some code setting up a, b, c, d
a += b;
c += d;
```

In sequential computing, a multiple-issue processor core has the hardware chops to issue more than one instruction per cycle. This is another way to speed up execution. For example, in the code above, there is no dependency between updating a and c. Multiple-Issue can be done statically (predefined) or dynamically (determined at run time).

![](<../../.gitbook/assets/Screen Shot 2021-01-29 at 5.04.23 PM.png>)

A chip that is capable of doing multiple-issue is also called a superscalar architecture. Title card!

## ILP to TLP

![Various ILP techniques](<../../.gitbook/assets/Screen Shot 2021-01-29 at 5.16.27 PM.png>)

To wrap up, pipelining, OOOE, and multiple-issue are techniques for Instruction-Level Parallelism (ILP). These techniques work within one thread, and we can get more optimizations by going up to the thread level, TLP, where a chip executes simultaneously from different processes or different threads. Note that at this point, we are still talking about parallelism within one core, not multicore.

## HTT

* HTT is an example implementation of TLP.
* The scheduler tries to issue instructions from both processes at the same time.
* HTT allows the OS to see one physical chip as two virtual chips.
* HTT is particularly useful when running simultaneous modestly demanding processes. In HPC, if one stream of instruction saturates the memory bandwidth, then it's less useful.

![](<../../.gitbook/assets/Screen Shot 2021-01-29 at 5.38.47 PM.png>)

![When one thread stalls (due to cache miss, branch mispredict, pipeline bubbles, etc.), the other thread chimes in at the same rate as a single thread running on the core](<../../.gitbook/assets/Screen Shot 2021-01-29 at 5.39.02 PM.png>)

A taxonomy for multi-threading is (Hennessey & Patterson):

* Coarse-grain multi-threading
* Fine-grain multi-threading
* Simultaneous multi-threading

To wrap up superscalar vs. TLP:

* Superscalar: Instructions associated with one PC
  * HW allows more than one instruction per cycle
  * One thread
* TLP: Instructions associated with two PCs
  * Processor handles instructions from different threads/processes
