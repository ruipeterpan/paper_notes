# Lecture 1: Course Overview

## Course Description

This course seeks to: 

1. Provide an overview of various advanced computing software and hardware solutions
2. Introduce **CUDA** for parallel computing on the Graphics Processing Unit \(GPU\)
3. Introduce the **OpenMP** solution to enabling parallelism across multiple CPU cores
4. Introduce the Message Passing Interface \(**MPI**\) standard for leveraging parallelism on a CPU cluster
5. Promote an understanding instrumental in deciding what parallel computing model is suitable for which problems.

## Linux "module" utility

[Linux man page](https://linux.die.net/man/1/module)

{% code title="Linux module usage" %}
```bash
[dan@euler ~]$ gcc --version
gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-16)
...
[dan@euler ~]$ module load gcc/6.4.0
[dan@euler ~]$ gcc --version
gcc (GCC) 6.4.0
...

[dan@euler ~]$ nvcc main.cu -o cudaprogram
bash: nvcc: command not found
[dan@euler ~]$ module avail cuda

--------------/usr/local/share/modulefiles ---------------------
cuda/0_user/cuda  cuda/7.5  cuda/8-rc  cuda/9    cuda/9.1  
cuda/7            cuda/8    cuda/8.0   cuda/9.0

[dan@euler ~]$ module load cuda/9
[dan@euler ~]$ nvcc main.cu -o cudaprogram

[dan@euler ~]$ module list
Currently Loaded Modulefiles:
1) gcc/6.4.0   2) gcc/0_cuda/6.4.0   3) cuda/9
[dan@euler ~]$ module unload cuda gcc
[dan@euler ~]$ nvcc
bash: nvcc: command not found
```
{% endcode %}

## The Euler cluster

* Files on the Euler remote cluster can be easily edited using the [Remote-SSH plugin for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)

## Slurm \(Simple Linux Utility for Resource Management\)

Slurm is used on Euler for job management and scheduling.

Slurm usage \(SBATCH flags documentation\) can be found [here](https://slurm.schedmd.com/sbatch.html).

{% code title="Example of a Slurm-specific batch script" %}
```bash
#!/usr/bin/env bash                     # intepret file as bash script
#SBATCH --job-name=HelloScript
#SBATCH-p wacc                          # a partition is a logical chunk of cluster
#SBATCH --time=0-00:00:10
#SBATCH --output=“hello_output-%j.txt”

#SBATCH --ntasks=1 --cpus-per-task=1    # simple jobs: one core suffices
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8             # for mpi
#SBATCH --cpus-per-task=4               # multithreaded jobs

#SBATCH --gres=gpu:1                    # gres: Generic RESource
                                        # --gres=type[:model]:N
                                        # e.g., gpu:gtx1080:3or infiniband:1                                        
#SBATCH --constraint=haswell


# regular bash script
cd $SLURM_SUBMIT_DIR                    # directory where the script is submitted from

name_str=“World”
echo “Hello, $name_str!”
```
{% endcode %}

{% code title="sbatch \(Slurm batch\) usage" %}
```bash
[dan@euler~]$ sbatch hello_slurm.sh     # submit a scheduling script to Slurm
Submitted batch job 1975385
[dan@euler~]$ cat hello_output-1975385.txt
Hello, World!
[dan@euler~]$
```
{% endcode %}

