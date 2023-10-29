# Venti: A New Approach to Archival Storage

## One-line Summary

Venti is a “building block” for constructing applications that frequently require an efficient, reliable, and accessible backup mechanism. The main technique for deduplication is to use hashing to identify different blocks.

## Paper Structure Outline

1. Introduction
2. Background
3. The Venti Archival Server
   1. Choice of Hash Function
   2. Choice of Storage Technology
4. Applications
   1. Vac
   2. Physical backup
   3. Plan 9 File system
5. Implementation
6. Performance
7. Reliability and Recovery
8. Related Work
9. Future Work
10. Conclusion

## Background & Motivation

The goal of Venti, a block-level network storage system, is to provide a **write-once archival** repository that can be shared by multiple client machines and applications.

## Design and Implementation

The interface of the system is a simple protocol that enables client applications to read and write variable-sized blocks of data. The primary technique is to use hashing on the data blocks to generate a fingerprint for identification. Some results are:

* Collision resistant: By using a good hash function, it is possible to consider the hash of a data block as **unique**.
* A block cannot be modified w/o changing its address: The behavior is intrinsically **write-once**. The write-once policy provides **data protection**, as no data will be overwritten.
* Writes are idempotent: Multiple writes of the same data can be coalesced and do not require additional storage space (**no extra space overhead**). Duplicate blocks will be discarded and **only one copy of the data will be retained**.
* The fingerprint allows Venti to provide inherent **integrity checking** of data. This allows the client to avoid errors from undetected data corruption and enables the server to identify when error recovery is necessary.

### Fingerprint/Hashing

The authors want Venti to employ a cryptographic hash function. For such a function, it is computationally infeasible to find two distinct inputs that hash to the same value. Venti utilizes the SHA-1 algorithm, which generates a 20-byte hash value, making the collision probability extremely low.&#x20;

### Applications (that work on top of Venti)

Applications are responsible for mapping the namespace to fingerprints. One approach to record the fingerprints is to pack them into additional blocks (pointer blocks) that are also written to the server. This approach can be applied recursively until a single fingerprint is obtained. The paper didn't really go deep into this. Some other work includes [sfsro](https://spqrlab1.github.io/papers/sfsro-tocs.pdf) (making a single-byte file system read-only using [Merkle trees](https://en.wikipedia.org/wiki/Merkle\_tree)).

![](<../../.gitbook/assets/Screen Shot 2020-12-28 at 11.23.42 PM.png>)

### Venti Implementation

![The data is append-only log on a RAID array.](<../../.gitbook/assets/Screen Shot 2020-12-28 at 11.27.47 PM.png>)

When a fingerprint comes in, a lookup on an on-disk index is performed to check where the data is stored (horrible performance due to random lookups of random fingerprints with no locality on disk). The caches are used to leverage the terrible performance, it's still not that good though (a lot of room for improvements, see the [Data Domain](data-domain-avoiding-the-disk-bottleneck-in-the-data-domain-deduplication-file-system.md) paper for some follow-ups).

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2020-12-28 at 11.32.35 PM.png>)

![Effects of Venti on data reduction](<../../.gitbook/assets/Screen Shot 2020-12-28 at 11.35.00 PM.png>)

## New Vocabulary

* Idempotent: The property of certain operations in mathematics and computer science whereby they can be applied multiple times without changing the result beyond the initial application.

## Links

* [Paper PDF](https://www.usenix.org/legacy/publications/library/proceedings/fast02/quinlan/quinlan.pdf)
* Thanks to Brian Chang for the review notes!

{% file src="../../.gitbook/assets/Venti.pdf" %}
Prof. Andrea Arpaci-Dusseau's course notes on Venti
{% endfile %}
