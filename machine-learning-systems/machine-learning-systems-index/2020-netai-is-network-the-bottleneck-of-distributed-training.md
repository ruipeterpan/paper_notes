# Is Network the Bottleneck of Distributed Training?

## Summary

Distributed training suffers from sub-linear scale-out. The authors argue that this is due to the network not being fully saturated as a result of the poor implementation of network transport. If the network can be fully utilized, distributed training can achieve an almost-linear scale-out. Also, in a highly-utilized network, the extent of gradient compression does not need to be that high.

## Background & Motivation

Current distributed DNN training using data parallelism suffers from a sub-linear scaling when scaled out. People have been optimizing the communication phase of distributed training, as the computation phase (the other phase) is embarrassingly parallel and should scale almost linearly. An example of those optimizations is gradient compression, which lies at the application level.

![](<../../.gitbook/assets/Screen Shot 2022-02-04 at 12.27.37 PM.png>)

The authors of this paper instead look at the network layer (network-level optimizations do not require changes at the application level).

## Evaluation

The authors first argue that the computation phase is not the bottleneck. They found that due to (1) distributed backward pass overlaps with all-reduce and (2) Horovod injects per-layer hooks in distributed training, the computation time has a slight increase in distributed training. However, this inevitable side effect (at most 15%) does not offset the extent (merely 56% - 75%) of the sub-linear scaling.

![](<../../.gitbook/assets/Screen Shot 2022-02-04 at 12.28.06 PM.png>)

Then, the only possibility is that the communication phase is the bottleneck, so the authors tried different network bandwidths. Surprisingly, the scaling factor line plateaus after 25Gbps, meaning that a faster network does not necessarily benefit the system.

![](<../../.gitbook/assets/Screen Shot 2022-02-04 at 12.08.08 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-02-04 at 12.14.03 PM.png>)The low network utilization at a high bandwidth explains the issue, as only a small fraction of the bandwidth is properly utilized. This might be explained by TCP being CPU-intensive at high speed (100 Gbps), but modern GPU instances have sufficient CPUs, and the authors found the actual CPU utilization is low. **The conclusion is that the poor implementation of network transport cannot fully saturate the available bandwidth during communication.**

![](<../../.gitbook/assets/Screen Shot 2022-02-04 at 12.30.50 PM.png>)A what-if simulation analysis shows that under a fully-utilized network, almost-linear scale-out is possible.

![](<../../.gitbook/assets/Screen Shot 2022-02-04 at 12.31.47 PM.png>)Also, gradient compression is useful in low-speed networks, but a large compression ratio is not necessary.

## Links & References

* [Paper PDF](https://dl.acm.org/doi/pdf/10.1145/3405671.3405810)
* [training-bottleneck on GitHub](https://github.com/netx-repo/training-bottleneck)
