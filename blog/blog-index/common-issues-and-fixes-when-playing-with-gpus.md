# Common Issues \(& Fixes\) when Playing with GPUs

### GPU Monitoring

* NVIDIA System Management Interface \(nvidia-smi\)
  * Use `watch -d -n 0.1 nvidia-smi` to eyeball the GPU usage. `-d` highlights the differences between successive updates, while `-n` specifies an update interval.
* [gpustat on GitHub](https://github.com/wookayin/gpustat)
* [dstat](https://www.tecmint.com/dstat-monitor-linux-server-performance-process-memory-network) \(for CPUs only\)

### CUDA

* [NVIDIA NVML Driver/library version mismatch](https://stackoverflow.com/q/43022843/9601555).
* [Existing package manager installation of the driver found \(Purging old versions of CUDA\)](https://askubuntu.com/a/1309386)
* [Different CUDA versions shown by nvcc and nvidia-smi](https://stackoverflow.com/q/53422407/9601555)

### NVIDIA Visual Profiler \(nvprof & nvvp\)

* nvprof comes with the CUDA Toolkit, so the easiest way to install this w/o any potential version mismatch issues is to install the CUDA Toolkit. This [gist](https://gist.github.com/sonots/5abc0bccec2010ac69ff74788b265086) is a great example of a workflow while using nvvp. 
* nvprof not found: Check the [doc about CUDA Toolkit post-installation actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions). In short, make sure that `PATH` and `LD_LIBRARY_PATH` are correctly updated. In general, nvprof should be available at `/usr/local/cuda-11.0/bin/nvprof`, while nvvp is installed at `/usr/local/cuda-11.0/libnvvp/nvvp`. The NVIDIA official documentation is bulky and looks scary, but with the benefit of hindsight, most solutions can be found in there.
* [The visual profiler requires a Java Runtime Environment \(JRE\)](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#setup-jre). To set this up, follow the [instructions here](https://forums.developer.nvidia.com/t/java-error-on-launching-visual-profiler-on-10-1-update-2/80384/4?u=rpan33).
* [CUPTI DLL not on your PATH environment](https://forums.developer.nvidia.com/t/nvidia-visual-profiler-is-unable-to-profile-application/108688/2?u=rpan33)/nvprof unable to locate CUPTI library. Adding the path to PATH: `export PATH=/usr/local/cuda-11.0/extras/CUPTI/lib64${PATH:+:${PATH}}`
* ["Workspace in use or cannot be created, choose a different one"](https://stackoverflow.com/q/7465793/9601555).

### CUDA Multi-Process Service \(MPS\)

* [Official documentation](https://docs.nvidia.com/deploy/mps/index.html#topic_2). Again, the documentation is bulky yet extensive and very helpful in retrospect.
* [Server](https://docs.nvidia.com/deploy/mps/index.html#topic_4_3_1), [Clients](https://docs.nvidia.com/deploy/mps/index.html#topic_4_3_2)
* The MPS servers are controlled by the MPS control daemon. Some utilities include [these](https://docs.nvidia.com/deploy/mps/index.html#topic_5_1_1). Note that once a daemon is started, it will only launch an MPS server if an MPS client connects to the control daemon and there is no server active.

