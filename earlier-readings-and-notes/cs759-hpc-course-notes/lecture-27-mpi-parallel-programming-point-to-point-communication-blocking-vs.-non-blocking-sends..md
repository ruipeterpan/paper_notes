# Lecture 27: MPI Parallel Programming Point-to-Point communication: Blocking vs. Non-blocking sends.

## Lecture Summary

* Last time
  * HPC via MPI
  * MPI point-to-point communication: The blocking flavor
* Today
  * Wrap up point-to-point communication
  * Collective communication

## Point-to-point communication

* Different "send" modes:
  * Synchronous send: MPI\_SSEND
    * Risk of deadlock/waiting -> idle time
    * High latency but better bandwidth than bsend
  * Buffered (async) send: MPI\_BSEND
    * Low latency/bandwidth
  * Standard send: MPI\_SEND
    * Up to the MPI implementation to device whether to do rendezvous or eager
    * Less overhead if in eager mode
    * Blocks in rendezvous, switches to sync mode
  * Ready send: MPI\_RSEND
    * Works only if the matching receive has been posted
    * Rarely used, very dangerous
* Receiving, all modes: MPI\_RECV
* Buffered send
  * Reduces overhead associated with data transmission
  * Relies on the existence of a buffer. Buffering incurs an extra memory copy&#x20;
  * Return from an MPI\_Bsend does not guarantee the message was sent: the message remains in the buffer until a matching receive is posted

![Blocking options](<../../.gitbook/assets/Screen Shot 2021-04-10 at 6.11.46 PM.png>)

![Deadlocks](<../../.gitbook/assets/Screen Shot 2021-04-10 at 6.15.29 PM.png>)

## Non-blocking point-to-point

* Blocking send: Covered above. Upon return from a send, you can modify the content of the buffer in which you stored data to be sent since the data has been sent
* Non-blocking send: The sender returns immediately, no guarantee that the data has been transmitted
  * Routine name starts with MPI\_I
  * Gets to do useful work (overlap communication with execution) upon return from the non-blocking call
  * Use synchronization call to wait for communication to complete
* MPI\_Wait: Blocks until a certain request is completed
  * Wait for multiple sends: Waitall, Waitany, Waitsome
* MPI\_Test: Non-blocking, returns quickly with status information
  * int MPI\_Test(MPI\_Request \*request, int \*flag, MPI\_Status \*status);
* MPI\_Probe: Allows for incoming messages to be queried prior to receiving them

![](<../../.gitbook/assets/Screen Shot 2021-04-10 at 6.20.21 PM.png>)

## Collective communications

* Three types of collective actions:
  * Synchronization (barrier)
  * Communication (e.g., broadcast)
  * Operation (e.g., reduce)
* [Writing distributed applications with PyTorch](https://pytorch.org/tutorials/intermediate/dist\_tuto.html) is a good tutorial
* Broadcast: MPI\_Bcast
* Gather: MPI\_Gather
* Scatter: MPI\_Scatter
* Reduce: MPI\_Reduce
  * Result is collected by the root only
* Allreduce: MPI\_Allreduce
  * Result is sent out to all ranks in the communicator
* Prefix scan: MPI\_Scan
* User-defined reduction operations: Register using MPI\_Op\_create()

![Visualization of the operations, excerpted from the Distributed PyTorch documentation](<../../.gitbook/assets/Screen Shot 2021-04-10 at 6.28.36 PM.png>)

![Predefined reduction operations](<../../.gitbook/assets/Screen Shot 2021-04-10 at 6.31.06 PM.png>)
