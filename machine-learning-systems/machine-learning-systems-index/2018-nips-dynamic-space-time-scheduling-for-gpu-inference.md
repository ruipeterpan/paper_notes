# \[2018 NIPS] Dynamic Space-Time Scheduling for GPU Inference

## Summary

The authors evaluated different multiplexing (time & space) techniques for ML inferences on GPUs and proposed ideas to achieve the best tradeoff across criterias.

## Background & Motivation

Almost all cloud inference service providers/frameworks assign each model an exclusive GPU. This, combined with the small batch sizes used in an online setting, results in low hardware utilization. Current approaches that multiplex workloads have different tradeoffs, and there is no single solution that wins on all criteria.

| Approach                                   | Utilization | Performance (throughput/latency) | Predictability/Performance Isolation |
| ------------------------------------------ | ----------- | -------------------------------- | ------------------------------------ |
| Exclusive access                           | Poor        | Good                             | Good                                 |
| Time multiplexing (CUDA context switching) | Average     | Poor                             | Good                                 |
| Spatial multiplexing                       | Good        | Average                          | Poor                                 |

## Design & Implementation

![](<../../.gitbook/assets/Screen Shot 2022-06-30 at 4.38.34 PM.png>)

The authors proposed software-level fusion of kernel operators across multiple inference jobs to get the best of all worlds.

## Links & References

* [Paper PDF](http://learningsys.org/nips18/assets/papers/102CameraReadySubmissionGPU\_Virtualization%20\(8\).pdf)
