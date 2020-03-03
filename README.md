# qrmethod_opencl_intel_fpga

We have implemented QR Decomposition based on modified Gram-Schmidt algorithm. Those files in this repository are OpenCL implementations for Intel FPGAs, they may work on other devices with some minor changes in kernel source code and major changes in the host source code.

**Naive QR Decomposition**
We used the standard algorithm not suitable for FPGA parallelism. That is the simplest algorithm found in the literature. We show three main version of this algorith, they are as follows:

- V1: Simple copy and paste from the original algorithm;
- V2: Inferred shift registers for accumulators;
- V3: Inferred accumulators from Arria 10 board, that is, we removed the shift registers (if you don't have an Arria 10, keep the shift registers, it is much faster for other older boards). We also used local memory to perform the computation, global memory is only used to read the initial data and write the results back.

Each version presents three variations for square root: native_sqrt, pow, and inverse square root. According to our results, sqrt and native_sqrt generate the same hardware in Intel boards. Inverse Square Root algorithm is available on: https://en.wikipedia.org/wiki/Fast_inverse_square_root , according to our results, that algorithm did not improve performance.


**Intel QR Decomposition**
We have implemented Intel QR Decomposition Algorithm for faster pipeline parallelism, we have renamed it to qr_article in this repository. This is an OpenCL version, the authors from the original algorithm used DSP Builder tools. We wanted to compare the performance of a hardware generated by a High Level Synthesis Tool with a manual implementation.

In this version, we did not implement the variation for square root. The standard functions seems generated the best hardware version regarding performance. In this folder, you are going to find float and int version of the algorithm.

The paper is available on https://ieeexplore.ieee.org/abstract/document/7856841 

**Compilation**
OpenCL for FPGAs requires more work than the usual OpenCL compilation process. First you must compile your kernel with the following command:

aoc -g -v spjacobi_single_no_global_write.cl

**tip:** use "--fp_relaxed" and "--fpc" (without quotes) for better performance, the first flag infers floating-point DSPs from Arria 10, and the second one relax the roundings from floating point operations. BE CAREFUL, that may cause some numerical error in the results. AOCX is the kernel binary after compilation.

After that, you can compile your host side with the follwing command:

icc -O3 -D__USE_XOPEN2K8 -DHAVE_CONFIG_H -DTESTB host.cpp -I/<path_to_quartus>/quartus_pro/16.0.2/hld/host/include -I/<path_to_project>/opencl_projects/QR_method/common/inc -L/<path_to_quartus>/quartus_pro/16.0.2/hld/linux64/lib -L/<path_to_BSP>/SR-5.0.2-OpenCL/opencl/opencl_bsp/host/linux64/lib -L<accelerator_abstraction_layer_path>/usr/local/lib -L/<path_to_quartus>/quartus_pro/16.0.2/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -lalterahalmmd -laltera_qpi_mmd -lrt -lltdl  -lpthread -lOSAL -lAAS -laalrt -lelf <path_to_project>/opencl_projects/QR_method/common/src/AOCLUtils/*.cpp -o program

**Execution**
You can execute it like any other executable binary, in our case (in Linux):

./program



