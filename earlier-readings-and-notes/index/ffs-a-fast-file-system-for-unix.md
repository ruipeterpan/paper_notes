# FFS: A Fast File System for UNIX

## One-line Summary

As a reimplementation of the UNIX file system, FFS addresses the problem of low data throughput rates/bandwidth usage by making the file system "disk aware".

## Paper Structure Outline

1. Introduction
2. Old file system
3. New file system organization
   1. Optimizing storage utilization
   2. File system parameterization
   3. Layout policies
4. Performance
5. File system functional enhancements
   1. Long file names
   2. File locking
   3. Symbolic links
   4. Rename
   5. Quotas
6. Acknowledgments

## Background & Motivation

![The old UNIX file system.](<../../.gitbook/assets/Screen Shot 2020-12-26 at 9.16.01 AM.png>)

As a predecessor of FFS, the original UNIX operating system (very-simple file system, VSFS) was simple and easy-to-use. It had some major problems, though:

* The biggest problem was **terrible performance**: As measured in this paper, in some cases, the disk bandwidth utilization was a mere 3%. The throughput was also low.
* This was because of **low locality**: The old UNIX file system treated the disk like it was a random-access memory, so inodes and data blocks were scattered across the disk.
* Another problem was **file system fragmentation**: The free space was not carefully managed. As data come and go, the file system got fragmented in that the free list ended up pointing to blocks spread across the disk, so accessing a logically contiguous file required going back and forth across the disk. (This problem was originally solved by disk defragmentation tools.)
* The original **block size was too small** (512 bytes): A smaller size was good for minimizing internal fragmentation (waste within the block) but **bad for transfers** as each block required a positioning overhead.

To resolve these issues, the solution is to make the file system "disk aware".

## Design and Implementation

### Cylinder groups

![A disk is divided into several cylinder groups.](<../../.gitbook/assets/Screen Shot 2020-12-26 at 9.13.34 AM.png>)

![IRL, as disks hide details of their geometry from clients, modern file systems organize the drive into block groups, each of which is a consecutive portion of the disk's address space. Note that IRL each group will contain many more blocks.](<../../.gitbook/assets/Screen Shot 2020-12-26 at 9.15.09 AM.png>)

![What FFS keeps within a single cylinder group. ib and db are inote bitmap and data bitmap that tracks whether the inodes and data blocks of the group are allocated. Bitmaps replaces freelists as bitmaps are faster to update/lookup and are space efficient.](<../../.gitbook/assets/Screen Shot 2020-12-26 at 9.15.35 AM.png>)

By putting two files in the same group, FFS ensures that accessing one after the other will not result in long seeks across the disk.

The superblock is duplicated and stored in different cylinder groups with a rotated location/offset. This reduces the chance of data loss due to corruption of the superblock (top platter damage).

### Layout policies

The basic mantra is simple: Keep related stuff together and keep unrelated stuff far apart.

* Placing directories: FFS finds the cylinder group with a low number of allocated directories and a high number of free inodes.
* Placing files: First, data blocks of a file in the same group as its inode are allocated. Second, file inodes are allocated in the same cylinder group with other files within the same directory.

A problem is the large files exception: Single, large files (> 48KB) can fill nearly all of a group, while ideally none of the cylinder groups should be completely full. The solution is to redirect block allocation to a different cylinder group when a file exceeds 48KB and at every megabyte thereafter.&#x20;

{% hint style="info" %}
The first spill over point at 48 kilobytes is the point at which a file on a 4096 byte block file system first requires a single indirect block. This appears to be a natural first point at which to redirect block allocation. The other spillover points are chosen with the intent of forcing block allocation to be redirected when a file has used about 25% of the data blocks in a cylinder group. In observing the new file system in day to day use, the heuristics appear to work well in minimizing the number of completely filled cylinder groups.
{% endhint %}

### Larger blocks & fragments within blocks

The size of a file system block is increased from 512 bytes to 4096 bytes (4 KB). This improves performance because:

1. Each disk transfer access more data
2. More files can be described without the need to access indirect blocks (since the direct blocks now contain more data)

### File system parameterization (rotationally optimal block placement)

![Think about this: In sequential reads, FFS firstly reads block 0. By the time the read is finished, block 1 had rotated under the head and to get to block 1, we now need a full rotation. FFS resolves this by figuring out the specific performance parameters of the disk and use those to decide on the exact staggered layout scheme.](<../../.gitbook/assets/Screen Shot 2020-12-26 at 9.30.15 AM.png>)

## Evaluation

![FFS is faster for both reads and writes and the disk bandwidth (\~3% to 47%)](<../../.gitbook/assets/Screen Shot 2020-12-26 at 8.28.56 AM.png>)

FFS also introduced some neat file system functional enhancements that are routines of today's operating systems and likely help FFS gain a stronger user base:

* **Long file names**: File names can now be of nearly arbitrary length (previously: 8 characters. now: 255 characters)
* **File locking**: Programs can now apply advisory shared or exclusive locks at the granularity of files
* **Symbolic links**: Users can now create an "alias" to any other file or directory on a system and thus are much more flexible
* **File renaming**: Atomic `rename()` operation
* **Quotas**: Restricts the amount of file system resources (#inodes & #disk blocks) that a user can obtain

## New Vocabulary

* Hard locks & advisory locks: A hard lock is always enforced when a program tries to access a file, while an advisory lock is only applied when it is requested by a program.

## Links

* [Paper PDF](https://dsf.berkeley.edu/cs262/FFS-annotated.pdf)
* [FFS in OSTEP](http://pages.cs.wisc.edu/\~remzi/OSTEP/file-ffs.pdf)
* [FFS in CS 537 @ UW-Madison](http://pages.cs.wisc.edu/\~shivaram/cs537-sp20-notes/ffs/cs537-ffs-notes.pdf)

{% file src="../../.gitbook/assets/L2-FFS+LFS.pptx" %}
Prof. Andrea's slides on FFS and LFS
{% endfile %}

