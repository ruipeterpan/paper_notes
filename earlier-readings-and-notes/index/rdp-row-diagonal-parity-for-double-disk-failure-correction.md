# RDP: Row-Diagonal Parity for Double Disk Failure Correction

## One-line Summary

RDP is an algorithm that protects against double disk failures. RDP can be applied to RAID systems. RDP is also known as RAID-DP/RAID-6 (There are other RAID-6 approaches to handle two disk failures, but RDP is the most intuitive).

## Paper Structure Outline

1. Introduction
2. Related Work
3. Double Disk Failure Models and Analysis
4. Row-Diagonal Parity Algorithm
5. Proof of Correctness
6. Performance Analysis
7. Algorithm Extensions
8. Implementation Experience
9. Measured Performance
10. Conclusions
11. Acknowledgments

## Background & Motivation

There are two types of disk failures: Individual disks can fail by whole-disk failure, whereby all the data on the disk becomes temporarily or permanently inaccessible, or by media failure, whereby a small portion of the data on a disk becomes temporarily inaccessible. The previous RAID only considers whole-disk failures.

Multiple disk errors are likely: the authors gave a detailed analysis of why this is the case in section 3 (which I'm not going to get into).

## Design and Implementation

RDP is built on RAID-4 or RAID-5. In this paper, we will focus on RAID-4.

![In this case, p = 5. We have (p+1) disks and (p-1) data disks.](<../../.gitbook/assets/Screen Shot 2020-12-27 at 2.13.28 PM.png>)

XOR is still used for parity. The figure shows the diagonal of each block. In the example above, if we have whole-disk failures on data disks 1 and 3, the data can be easily recovered in many ways.

RDP can also be extended to encompass multiple RAID-4 or RAID-5 disk arrays in a single RDP disk array.

## Evaluation

* Read performance is unaffected.
* Sequential write: Write p-1 stripes at once for best performance (update row and diagonal parity at the same time).
* Partial stripe writes: Writing d blocks by subtraction requires 2d+4 I/Os (d+2 for read, d+2 for write), and writing d blocks by additive requires n I/Os (n-d-2 for read, d+2 for write). Thus, we use a combination of additive and subtractive.
* Proof of correctness and optimality is covered in the paper.

![Write performance measured: RDP gives a much better reliability for the same cost and performance.](<../../.gitbook/assets/Screen Shot 2020-12-27 at 2.22.53 PM.png>)

{% hint style="info" %}
* G: number of separate RAID groups connected to the filer
* d: number of data disks per RAID group
* p: number of parity disks per RAID group
{% endhint %}

## New Vocabulary

* NetApp: A cloud data services and data management company.

## Links

* [Paper PDF](https://www.usenix.org/legacy/publications/library/proceedings/fast04/tech/corbett/corbett.pdf)

{% file src="../../.gitbook/assets/L3+L4+L5-RAID+RDP+iBench.pptx" %}
Prof. Andrea's slides on RAID and RDP
{% endfile %}
