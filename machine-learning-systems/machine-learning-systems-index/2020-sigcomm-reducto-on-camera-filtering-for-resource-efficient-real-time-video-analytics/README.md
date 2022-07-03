# \[2020 SIGCOMM] Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics

## Summary

Frame differencing is an existing technique that improves the efficiency of video analytics pipelines. Reducto makes this approach more efficient by performing on-camera frame filtering using server-guided decisions to dynamically adjust the filtering threshold and smartly select the best differencing feature.

![](<../../../.gitbook/assets/Screen Shot 2022-02-07 at 12.12.11 PM.png>)

## Background & Motivation

In real-time video analytics pipelines, cameras send video streams to cloud servers, which then immediately run object detection models to answer user queries. Such pipelines aim for high accuracy and low latency.

Video analytics is resource-intensive (compute & network)! A technique to improve efficiency is frame filtering, which removes (i.e., do not send) frames that wouldn't change the query results. There are three existing approaches to this end:

1. Approximate model: Only send frames to DNN if the on-camera compressed model is not confident
   1. Con: Still too slow on camera
2. Specialized binary classifiers: Only send frames that contain certain objects
   1. Con: Misses filtering opportunities (objects are present but query results do not change across frames)
3. Pixel-level frame differencing: Only send frames if the low-level features like pixel values have changed drastically as we would expect a different result from the previous frames
   1. Cons: Existing approaches use single, static thresholds while the video content can be highly dynamic, so they cannot reliably meet accuracy targets. Moreover, they rely solely on pixel comparison, whereas there might be other low-level frame differences that are potentially more effective.

![](<../../../.gitbook/assets/Screen Shot 2022-02-07 at 12.23.29 PM.png>)

This work focuses on improving (3) by addressing two questions: (1) How do we dynamically determine the filtering threshold? (2) Which differencing feature should we use?

## Design & Implementation

### Dynamic threshold

![](<../../../.gitbook/assets/Screen Shot 2022-02-07 at 12.13.36 PM.png>)

* What's the threshold that filters the most frames while meeting the target accuracy?
* Using a small, unfiltered video, split the video into several segments and construct a hash table that maps the diff values to thresholds
* Server: builds the hash table (expensive)
* Camera: looks up in the hash table (cheap)

### Choosing the differencing feature

![](<../../../.gitbook/assets/Screen Shot 2022-02-07 at 12.22.47 PM.png>)

* Server: calculates the best feature (expensive)
* Run once per query type (e.g. counting)

### Overall pipeline

![](../../../.gitbook/assets/IMG\_1BAB9EE7A80E-1.jpeg)

![](<../../../.gitbook/assets/Screen Shot 2022-02-07 at 12.22.20 PM.png>)

## Evaluation

![](<../../../.gitbook/assets/Screen Shot 2022-02-07 at 12.25.46 PM.png>)

![](<../../../.gitbook/assets/Screen Shot 2022-02-07 at 12.28.44 PM.png>)

![](<../../../.gitbook/assets/Screen Shot 2022-02-07 at 12.28.57 PM.png>)

## Links & References

* [Paper PDF](https://www.cs.princeton.edu/\~ravian/publications/reducto\_sigcomm20.pdf)
* [Presentation video at SIGCOMM '20](https://www.youtube.com/watch?v=IllEKLVUiYM)
