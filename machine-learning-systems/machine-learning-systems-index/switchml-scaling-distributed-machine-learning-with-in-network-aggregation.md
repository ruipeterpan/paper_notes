# SwitchML: Scaling Distributed Machine Learning with In-Network Aggregation

## Summary

Modern distributed ML training is communication-intensive. Thanks to the corporate overlords, emerging hardware shows up for help. Programmable switches can aggregate model updates in-network, making the network itself an accelerator for ML.

## Background & Motivation

In recent years, we have seen orders of magnitude faster capability improvements in compute than networks. Furthermore, the ratio of communication to computation in the workload itself has shifted. As a result, in distributed training, the network is becoming the bottleneck.

A new approach for model updates is in-network aggregation. In this approach, workers send their model updates over the network, where an aggregation primitive in the network sums the updates and distributes only the resulting value. This offers a fundamental advantage over all-reduce and PS since it avoids end-host processing required to perform aggregation and therefore provides "sub-RTT" latency.

## Design & Implementation

The idea sounds amazing but it comes with challenges. First, switches' packet processing capabilities are limited, and ML uses floating-point values, while integer computing is the norm in programmable switches. Second, on-chip memory is also small (tens of MBs while model updates might have hundreds of megabytes of gradients). Finally, the system must be resilient to packet loss without impact on efficiency or correctness. To this end, the authors propose SwitchML which co-designs in-switch processing with an end-host transport layer and ML frameworks.

### SwitchML overview

* Combined switch-host architecture: The switch performs integer aggregation, while end hosts are responsible for managing reliability and performing more complex computations.
* Pool-based streaming aggregation: SwitchML streams aggregation through the switch. End hosts handle the management of aggregators in a pool, leaving the switch dataplane with a simple design.
* Fault-tolerant protocols: Recover from packet loss with minimal overheads & handles worker/network failures
* Quantized integer-based aggregation: Floating-point values are converted to 32-bit integers to satisfy the computing power of switches. This process is done at end hosts without impacting training accuracy.

![](<../../.gitbook/assets/Screen Shot 2022-02-16 at 12.04.51 PM.png>)

### Aggregation protocol

* Switch-side: A pool-based design addresses two limitations. First, it removes the need to store an entire model update on a switch at once. Second, it allows processing to be done at the packet level by performing the aggregation in small pieces, at most k integers at a time.
* Worker-side: After the initial batch of packets is sent, each worker only sends a new packet with the next piece of update once it has received the aggregated packets returned from the switch. This simple communication scheme does not require any explicit coordination among workers yet still achieves agreement on which slots to use.

### Packet loss

The natural way to deal with packet losses is retransmissions after timeouts. However, this naive approach has two main challenges: (1) differentiating packets that are lost on the upward paths vs. the downward ones, and (2) being able to retransmit an aggregated response that is lost on the way back to a worker. The solutions are (1) explicitly maintaining information as to which workers have already contributed updates to a given slot to ignore duplicate transmissions, and (2) maintaining a shadow copy of the previous result for each slot, which allows the switch to retransmit a dropped result packet for a slot even when the switch has started reusing the slot for the next chunk. This ensures that no worker node can ever lag more than one chunk behind any of the others for a particular slot.

### Quantizing floating-point values

SwitchML uses a numeric representation, inspired by block floating-point, that combines 32-bit fixed-point addition in the switch with adaptive scaling on the workers. This representation is used only when aggregating gradients. Empirically, this does not hurt convergence.

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2022-02-16 at 1.50.35 PM.png>)

## Links & References

* [Paper PDF](https://www.usenix.org/system/files/nsdi21-sapio.pdf)
* [Presentation video at NSDI '21](https://www.youtube.com/watch?v=FIZsXfeZrvE)
* [Presentation slides at NSDI '21](https://www.usenix.org/system/files/nsdi21\_slides\_sapio.pdf)
* [SwitchML on GitHub](https://github.com/p4lang/p4app-switchML)
