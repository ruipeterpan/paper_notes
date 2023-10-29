# \[2018 NIPS] Dynamic Space-Time Scheduling for GPU Inference

## Summary

The authors evaluated different multiplexing (time & space) techniques for ML inferences on GPUs and proposed ideas to achieve the best tradeoff across criterias.

## Background & Motivation

Almost all cloud inference service providers/frameworks assign each model an exclusive GPU. This, combined with the small batch sizes used in an online setting, results in low hardware utilization. Current approaches that multiplex workloads have different tradeoffs, and there is no single solution that wins on all criteria.

<table><thead><tr><th>Approach</th><th width="150">Utilization</th><th width="179.18672199170126">Performance (throughput/latency)</th><th width="223">Predictability/Performance Isolation</th></tr></thead><tbody><tr><td>Exclusive access</td><td>Poor</td><td>Good</td><td>Good</td></tr><tr><td>Time multiplexing (CUDA context switching)</td><td>Average</td><td>Poor</td><td>Good</td></tr><tr><td>Spatial multiplexing</td><td>Good</td><td>Average</td><td>Poor</td></tr></tbody></table>

## Design & Implementation

![](<../../.gitbook/assets/Screen Shot 2022-06-30 at 4.38.34 PM.png>)

The authors proposed software-level fusion of kernel operators across multiple inference jobs to get the best of all worlds.

## Links & References

* [Paper PDF](http://learningsys.org/nips18/assets/papers/102CameraReadySubmissionGPU\_Virtualization%20\(8\).pdf)
