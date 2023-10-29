# Lecture 2: From Code to Instructions. The FDX Cycle. Instruction Level Parallelism.

## Lecture Summary

This class is basically a recap of an "Intro to Machine Organization" class. The topics covered are instruction, assembly code, registers, CISC vs. RISC, CPU organization (CU/ALU), FDX cyle.

[CS 61C @ Berkeley (Su19)](https://inst.eecs.berkeley.edu/\~cs61c/su19/) is a great class with many more in-depth slides on these topics.

## CPU Organization

![A (somewhat simplified) schematic architecture](<../../.gitbook/assets/Screen Shot 2021-01-27 at 11.23.15 AM.png>)

The Control Unit (CU) controls the "datapath" (i.e., the hardware collection of functional units + registers + data buses), while the Arithmetic Logic Unit (ALU) executes arithmetic and load/store operations.

## From C to Machine Code

![C code -> intermediate representation -> assembly code -> machine code/instructions](<../../.gitbook/assets/Screen Shot 2021-01-27 at 11.32.57 AM.png>)

The same C code leads to different assembly code using different ISAs and even using different flags during compilation. An ISA (Instruction Set Architecture) is a set of commands (e.g., sw, addiu, lw) that the CU understands. The two paradigms for ISAs are RISC (Reduced Instruction Set Computing Architecture) and CISC (Complex). The major difference is that in RISC, an instruction is encoded into a fixed set of bits (64), while CISC (e.g., Intel/AMD x86) instructions have various lengths.

### The FDX (Fetch-Decode-Execute) Cycle

* Fetch: An instruction is fetched from memory
* Decode: The string of 1s and 0s are decoded by the CU. Example: [RISC-V Green Card](https://www.cl.cam.ac.uk/teaching/1617/ECAD+Arch/files/docs/RISCVGreenCardv8-20151013.pdf)
* Execute: Once all data (operands) available, instruction is executed

![Integrated Circuits: From Transistors to Chip Microarchitecture](<../../.gitbook/assets/Screen Shot 2021-01-27 at 12.10.48 PM.png>)
