# Big Data Systems Papers - Short Notes

## \[2012 HotNets] Coflow: A Networking Abstraction for Cluster Applications

Coflows make it easier for applications to convey their communicatino semantics to the network, which in turn enables the network to better optimize common communication patterns.

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.10.07 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.10.36 PM.png>)

## \[2014 SIGCOMM] Efficient coflow scheduling with Varys

* Smallest-Effective-Bottleneck-First (SEBF) heuristic: greedily schedules a coflow based on its bottleneck’s completion time
* Minimum-Allocation-for-Desired-Duration (MADD) algorithm: slows down all the flows in a coflow to match the completion time of the flow that will take the longest to finish, so that other coexisting coflows can make progress

<img src="../../.gitbook/assets/Screen Shot 2022-07-11 at 2.12.09 PM.png" alt="" data-size="original">

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.12.44 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.13.18 PM.png>)

## \[2015 SIGCOMM] Aalo: Efficient coflow scheduling without prior knowledge

* Removed the clairvoyance requirements of the coflow information
* Aalo employs Discretized Coflow-Aware Least-Attained Service (D-CLAS) to separate coflows into a small number of priority queues based on how much they have already sent across the cluster. By performing prioritization across queues and by scheduling coflows in the FIFO order within each queue, Aalo’s non-clairvoyant scheduler reduces coflow completion times while guaranteeing starvation freedom.

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.15.34 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.15.57 PM.png>)

