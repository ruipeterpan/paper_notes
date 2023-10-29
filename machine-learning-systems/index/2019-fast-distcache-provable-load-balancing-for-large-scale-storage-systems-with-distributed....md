---
description: ...Caching
---

# \[2019 FAST] DistCache: Provable Load Balancing for Large-Scale Storage Systems with Distributed...

## Summary

This paper presents a new distributed caching mechanism for addressing load imbalance in large-scale storage systems.

![](<../../.gitbook/assets/Screen Shot 2022-07-03 at 10.41.36 PM.png>)

## Background & Motivation

Cloud service providers use large clusters to store data. The data access workload is skewed (power law distribution), which creates load imbalance, resulting in low throughput and long tail latencies. The objective is to achieve load balancing in distributed storage systems.

![](<../../.gitbook/assets/IMG\_49EC455512C4-1 (1).jpeg>)A common approach is to add a front-end cache node as a load balancer.&#x20;

![](<../../.gitbook/assets/Screen Shot 2022-07-03 at 10.04.17 PM.png>)The problem is that nowadays, cloud-scale distributed storage spans across many clusters, which exposes scalability issues. Given that the cache throughput is 10-100 times of the server throughput, one caching node (e.g., a switch) can only guarantee load balancing for 10-100 servers (a few racks of servers within a cluster). In other words, a single cache node only guarantees intra-cluster load balancing, not inter-cluster load balancing.

![](<../../.gitbook/assets/Screen Shot 2022-07-03 at 10.07.36 PM.png>)Adding one cache node as the load balancer within each cluster also doesn't work: between clusters, load imbalance still exists. Adding another cache node atop all the per-cluster cache nodes does not work due to the throughput constraint.

Thus, we need a layer of distributed caching as the load balancer.

## Design & Implementation

Some key design choices include:

* Cache allocation with independent hash functions: The intuition is that if one cache node in a layer is overloaded by receiving too many queries to its cached objects, because the hash functions of the two layers are independent, the set of hot objects would be distributed to multiple cache nodes in another layer with high probability.
* Query routing with the power-of-two-choices: The sender of a query looks at the loads of the cache nodes that cache the queried object and sends the query to the less-loaded node.

![](<../../.gitbook/assets/Screen Shot 2022-07-03 at 10.34.42 PM.png>)

These mechanisms can be applied recursively for multi-layer hierarchical caching.

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2022-07-03 at 10.32.07 PM.png>)

## Links & References

* [Paper PDF](https://www.usenix.org/system/files/fast19-liu.pdf)
* [Presentation video at FAST '19](https://www.youtube.com/watch?v=iLsBC1yjH40)
* [Presentation slides at FAST '19](https://www.usenix.org/sites/default/files/conference/protected-files/fast19\_slides\_liu.pdf)
