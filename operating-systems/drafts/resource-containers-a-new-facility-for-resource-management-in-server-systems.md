# Resource containers: A new facility for resource management in server systems 

### One-liner Summary

A resource container is an operating systems abstraction that separates the notion of a protection domain from a resource principal. It allows for fine-grained resource management.

### Paper Structure Outline

1. Introduction
2. Typical models for high-performance servers
3. Shortcomings of current resource management models
   1. The distinction between shceduling entities and activities
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
   7. Immunitiy against SYN-flooding
   8. Isolation of virtual servers
6. Related Work
7. Conclusion

### Background & Motivation

In resource management, current systems do not separate the notion of "protection domain" \(where accounting of the tasks is done\) and "resource principal" \(where actual work gets performed\). Also, an application does not have control over how much resources the kernel consumes on behalf of the application. In this work, the two notions are separated by the new operating systems abstraction, resource containers.

### Design and Implementation

### Evaluation

### New Vocabulary

* 
### Links

* Paper PDF
* Presentation Video at xxx
* Presentation Slides
* * xxx on GitHub
* 








