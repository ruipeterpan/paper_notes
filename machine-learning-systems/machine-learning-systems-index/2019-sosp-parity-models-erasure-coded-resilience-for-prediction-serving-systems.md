# \[2019 SOSP] Parity Models: Erasure-Coded Resilience for Prediction Serving Systems

## Summary

This work uses erasure codes for reducing tail latency in ML inference.

![](<../../.gitbook/assets/Screen Shot 2022-07-10 at 9.53.05 PM.png>)

## Background & Motivation

ML inference, typically done in large-scale clusters, is latency-sensitive. However, slowdowns (network/compute contention) and failures in clusters might cause inference queries to miss their SLOs. This work aims to alleviate the effects of slowdowns and failures to reduce tail latency.

Erasure codes is a technique widely deployed in systems (e.g., storage systems, communication systems) for resource-efficient data corruption prevention. The difference between erasure codes for ML serving and for traditional settings is the need to handle computation over inputs. In other words, the encoding and decoding must hold over computation F.&#x20;

![](<../../.gitbook/assets/Screen Shot 2022-07-10 at 9.55.51 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-07-10 at 9.57.23 PM.png>)

The problem boils down to: How do we design the erasure codes for ML inference?

## Design & Implementation

Current approaches hand-craft erasure codes, which is relatively straightforward for a linear computation F, but is far more challenging for non-linear computations like ML serving. The authors overcome this challenge by taking a learning-based approach. Slap a NN, problem solved!

![](<../../.gitbook/assets/Screen Shot 2022-07-10 at 10.03.14 PM.png>)

But wait! Using NNs for encoders/decoders is computationally expensive. Instead, the authors use simple, fast encoders/decoders and operate over parities using a new computation model, namely the parity model. In this diagram, the parity model takes as input parity queries P = X1 + X2 and outputs Fp(P) = F(X1) + F(X2), which can later be used to reconstruct F(X2).&#x20;

![](<../../.gitbook/assets/Screen Shot 2022-07-10 at 10.04.23 PM.png>)

What a brilliant idea. We can also tweak the settings of this process, e.g. using a larger degree of query multiplexing (erasure codes parameter), or using different encoders/decoders instead of the simple summation encoder.&#x20;

![For example, for image tasks, we can downsample multiple queries and concatenate them into a single query](<../../.gitbook/assets/Screen Shot 2022-07-10 at 10.20.50 PM.png>)

## Evaluation

Note that although there is an accuracy loss, the inaccuracy only comes into play when predictions are otherwise slowed down or straigh up failed, which violate the latency requirements. This still sounds like a pretty good tradeoff, although I am curious about the accuracy loss on larger datasets and models with more complex architectures.

![Evaluation of the accuracy loss](<../../.gitbook/assets/Screen Shot 2022-07-10 at 10.18.03 PM.png>)

![Tail latency reduction in the presence of resource contention](<../../.gitbook/assets/Screen Shot 2022-07-10 at 10.23.23 PM.png>)

## Links & References

* [Paper PDF](https://www.cs.cmu.edu/\~rvinayak/papers/sosp2019parity-models.pdf)
* [Presentation video at SOSP '19](https://www.youtube.com/watch?v=NlrH\_4HNJI4) (one of my favorite talks)
* [parity-models on GitHub](https://github.com/Thesys-lab/parity-models)
