# Resource Containers: A New Facility for Resource Management in Server Systems

## One-line Summary

A resource container is an operating systems abstraction that separates the notion of a protection domain from a resource principal. It allows for fine-grained resource management.

## Paper Structure Outline

1. Introduction
2. Typical models for high-performance servers
3. Shortcomings of current resource management models
   1. The distinction between scheduling entities and activities
   2. Integrating network processing with resource management
   3. Consequences of misidentified resource principals
4. A new model for resource management
   1. Resource containers
   2. Containers, processes, and threads
   3. Resource containers and CPU scheduling
   4. Other resources
   5. The resource container hierarchy
   6. Operations on resource containers
   7. Kernel execution model
   8. The use of resource containers
5. Performance
   1. Prototype implementation
   2. Experimental environment
   3. Baseline throughput
   4. Costs of new primitives
   5. Prioritized handling of clients
   6. Controlling resource usage of CGI processing
   7. Immunity against SYN-flooding
   8. Isolation of virtual servers
6. Related Work
7. Conclusion

## Background & Motivation

In resource management, current systems do not separate the notion of "protection domain" \(where the accounting of the tasks is done\) and "resource principal" \(where actual work gets performed\). Processes and threads are both resource principals as well as protection domains. Also, an application does not have control over how much resources the kernel consumes on behalf of the application. In this work, the two notions are separated by the new operating systems abstraction, resource containers.

## Design and Implementation

## Evaluation

## New Vocabulary

* Resource principals: Entities for which separate resource allocation and accounting are done
* Protection domain: Entities that need to be isolated from each other

## Links

* [Paper PDF](https://www.usenix.org/legacy/publications/library/proceedings/osdi99/full_papers/banga/banga.pdf)
* [Lecture slides from EECS 443 @ Northwestern University](https://users.cs.northwestern.edu/~fabianb/classes/eecs-443-w09/lectures/RContainers.pdf)
* [Lecture slides from CS 5204 @ Virginia Polytechnic](http://courses.cs.vt.edu/~cs5204/fall14-butt/lectures/ResourceContainers.pdf)
* [Discussion panel from CS 736 @ UW-Madison](http://pages.cs.wisc.edu/~swift/classes/cs736-fa12/blog/2012/10/resource_containers_a_new_faci.html)

