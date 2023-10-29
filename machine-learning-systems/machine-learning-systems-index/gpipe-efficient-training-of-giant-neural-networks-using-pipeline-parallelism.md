# \[2019 NIPS] GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

## One-line Summary

GPipe presents pipeline parallelism on top of model parallelism for better hardware utilization.

## Paper Structure Outline

1. Introduction
2. The GPipe Library
   1. Interface
   2. Algorithm
   3. Performance Optimization
3. Performance Analyses
   1. Performance Overhead Breakdown
4. Image Classification
5. Massive Massively Multilingual Machine Translation
6. Design Features and Trade-Offs
7. Conclusion

## Background & Motivation

Generally, the larger (# parameters) a model is, the higher accuracy it yields. However, we are hitting the bottleneck on the memory of a single accelerator.&#x20;

![](<../../.gitbook/assets/Screen Shot 2021-05-15 at 4.39.20 PM.png>)

Traditional approaches to resolve this include:

* Recompute forward activations during the backprop calculations
  * Trades compute for memory: Activations are not stored but have to be recomputed
* Memory swap: Copy activations back to the CPU or main memory, and then copy them back
  * Communication between the CPU & accelerator becomes the bottleneck
* Parallelism: Split the computation between multiple "workers"
  * Data parallelism: Works well when there are a few parameters and lots of data and computations
  * Model parallelism: Works well when the number of parameters is large compared to data

## Design and Implementation

Vanilla model parallelism is not time efficient because of the serialized dependencies, and it also leads to system underutilization (bubbles between compute blocks).

GPipe uses pipeline parallelism to integrate data and model parallelism by dividing a minibatch into smaller microbatches so that accelerators can operate in parallel on different microbatches.

![](<../../.gitbook/assets/Screen Shot 2021-05-15 at 4.40.08 PM.png>)

The user is required to define (1) the number of model partitions, (2) the number of micro-batches, and (3) the sequence/definition of the layers that define the model.&#x20;

GPipe uses re-materialization to reduce the activation memory requirements. During the forward pass, only the output activations at the partition boundaries are stored. During the backward pass, the composite forward function is recomputed at each accelerator. The authors found that the bubble overhead (idle time on every accelerator) is negligible when M, the number of micro-steps, is bigger than 4 \* K, the number of accelerators, as the recomputations during the backward pass can be scheduled w/o waiting for the gradients from earlier layers.

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2021-05-15 at 5.16.52 PM.png>)

GPipe scaled up AmoebaNet in both the number of channels and the size of the input image. The giant models report competitive results on all target datasets.

![](<../../.gitbook/assets/Screen Shot 2021-05-15 at 5.17.01 PM.png>)

![Overhead breakdown](<../../.gitbook/assets/Screen Shot 2021-05-15 at 5.13.26 PM.png>)

## Links

* [Paper PDF](https://papers.nips.cc/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf)
* [Presentation video by Kartik Nanda](https://www.youtube.com/watch?v=9s2cum25Kkc)
* [A GPipe implementation in PyTorch on GitHub](https://github.com/kakaobrain/torchgpipe)
* [The integration of GPipe is done on tensorflow/lingvo](https://github.com/tensorflow/lingvo)
