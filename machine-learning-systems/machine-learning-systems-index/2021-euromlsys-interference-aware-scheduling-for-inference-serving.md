# Interference-Aware Scheduling for Inference Serving

## Summary

This work proposes a scheduler for inference workloads on heterogeneous hardware. The scheduler is aware of and proactive to interferences between co-located jobs, therefore outperforming baseline policies like lease-loaded.

## Background & Motivation

Inference serving schedulers co-locate models to improve resource utilization. However, the least-loaded scheduling policy, popular in the context of VM task scheduling, is agnostic to the interference/latency degradation created by co-location, thus yielding sub-optimal scheduling result.&#x20;

## Design & Implementation

![](<../../.gitbook/assets/Screen Shot 2022-06-30 at 2.24.04 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-06-30 at 2.29.56 PM.png>)

By using a unified predictor instead of maintaining separate predictors for different co-location degrees and machine types, we are able to (1) reduce the efforts needed to train multiple predictors and (2) exploit the similarity across co-location configurations (e.g., the same models on an 8vCPU VM vs. a 32vCPU VM).

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2022-06-30 at 2.37.57 PM.png>)

## Links & References

* [Paper PDF](https://dl.acm.org/doi/pdf/10.1145/3437984.3458837)
