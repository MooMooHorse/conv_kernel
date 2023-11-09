```
✱ Running bash -c "nsys profile --stats=true ./m2"   \\ Output will appear after run is complete.
**** collection configuration ****
        force-overwrite = false
        stop-on-exit = true
        export_sqlite = true
        stats = true
        capture-range = none
        stop-on-range-end = false
        Beta: ftrace events:
        ftrace-keep-user-config = false
        trace-GPU-context-switch = false
        delay = 0 seconds
        duration = 0 seconds
        kill = signal number 15
        inherit-environment = true
        show-output = true
        trace-fork-before-exec = false
        sample_cpu = true
        backtrace_method = LBR
        wait = all
        trace_cublas = false
        trace_cuda = true
        trace_cudnn = false
        trace_nvtx = true
        trace_mpi = false
        trace_openacc = false
        trace_vulkan = false
        trace_opengl = true
        trace_osrt = true
        osrt-threshold = 0 nanoseconds
        cudabacktrace = false
        cudabacktrace-threshold = 0 nanoseconds
        profile_processes = tree
        application command = ./m2
        application arguments = 
        application working directory = /build
        NVTX profiler range trigger = 
        NVTX profiler domain trigger = 
        environment variables:
        Collecting data...
Running test case 1
B = 1 M = 3 C = 3 H = 224 W = 224 K = 3 S = 1
Running test case 2
B = 2 M = 3 C = 3 H = 301 W = 301 K = 3 S = 2
Running test case 3
B = 3 M = 3 C = 3 H = 196 W = 196 K = 3 S = 3
Running test case 4
B = 4 M = 3 C = 3 H = 239 W = 239 K = 3 S = 4
All test cases passed
Test batch size: 10000
Loading fashion-mnist data...Done
Loading model...Done
Conv-GPU==
Layer Time: 650.689 ms
Op Time: 23.3266 ms
Conv-GPU==
Layer Time: 508.971 ms
Op Time: 81.0906 ms

Test Accuracy: 0.8714

        Generating the /build/report1.qdstrm file.
        Capturing raw events...

        **** WARNING: The collection generated 658618 total events. ****
        Importing this QDSTRM file into the NVIDIA Nsight Systems GUI may take several minutes to complete.

        Capturing symbol files...
        Saving diagnostics...
        Saving qdstrm file to disk...
        Finished saving file.


Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.

Importing...

Importing [==================================================100%]
Saving report to file "/build/report1.qdrep"
Report file saved.
Please discard the qdstrm file and use the qdrep file instead.

Removed /build/report1.qdstrm as it was successfully imported.
Please use the qdrep file instead.

Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.

Exporting 658517 events:

0%   10   20   30   40   50   60   70   80   90   100%
|----|----|----|----|----|----|----|----|----|----|
***************************************************

Exported successfully to
/build/report1.sqlite

Generating CUDA API Statistics...
CUDA API Statistics (nanoseconds)

Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
-------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
   73.8      1053104319          20      52655216.0           35314       585874693  cudaMemcpy                                                                      
   17.3       246980661          20      12349033.1            3155       242366955  cudaMalloc                                                                      
    7.3       104560603          10      10456060.3            3201        81063083  cudaDeviceSynchronize                                                           
    1.1        16299525          10       1629952.5           17321        16071973  cudaLaunchKernel                                                                
    0.4         6292996          20        314649.8            2758         3703429  cudaFree                                                                        




Generating CUDA Kernel Statistics...

Generating CUDA Memory Operation Statistics...
CUDA Kernel Statistics (nanoseconds)

Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
-------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
  100.0       104424277           6      17404046.2            9471        81061518  conv_forward_kernel                                                             
    0.0            2815           2          1407.5            1311            1504  do_not_remove_this_kernel                                                       
    0.0            2592           2          1296.0            1280            1312  prefn_marker_kernel                                                             


CUDA Memory Operation Statistics (nanoseconds)

Time(%)      Total Time  Operations         Average         Minimum         Maximum  Name                                                                            
-------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
   92.9       972281014           6     162046835.7           12672       584985875  [CUDA memcpy DtoH]                                                              
    7.1        73858071          14       5275576.5            1152        38832965  [CUDA memcpy HtoD]                                                              


CUDA Memory Operation Statistics (KiB)

            Total      Operations            Average            Minimum            Maximum  Name                                                                            
-----------------  --------------  -----------------  -----------------  -----------------  --------------------------------------------------------------------------------
        1723922.0               6           287320.4            148.535          1000000.0  [CUDA memcpy DtoH]                                                              
         545660.0              14            38975.7              0.004           288906.0  [CUDA memcpy HtoD]                                                              




Generating Operating System Runtime API Statistics...
Operating System Runtime API Statistics (nanoseconds)

Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
-------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
   33.3     98087295722         994      98679372.0           32159       100216016  sem_timedwait                                                                   
   33.3     97998954884         993      98689783.4           34860       100248202  poll                                                                            
   20.4     60019769253           2   30009884626.5     21764347434     38255421819  pthread_cond_wait                                                               
   12.9     38010984501          76     500144532.9       500090526       500167319  pthread_cond_timedwait                                                          
    0.1       194391747       13943         13941.9            1005        55608535  read                                                                            
    0.1       166743808         940        177387.0            1015        21310918  ioctl                                                                           
    0.0         3405353          98         34748.5            1484         1350026  mmap                                                                            
    0.0         1215336         101         12033.0            3901           45945  open64                                                                          
    0.0          322058          19         16950.4            3624           56830  fopen64                                                                         
    0.0          315438           5         63087.6           44593           94434  pthread_create                                                                  
    0.0          278797          26         10723.0            1494          209508  fopen                                                                           
    0.0          172814           3         57604.7           41471           88532  fgets                                                                           
    0.0          105395          18          5855.3            1331           13581  munmap                                                                          
    0.0           92156          15          6143.7            2571           10651  fflush                                                                          
    0.0           82211          15          5480.7            4297           12613  write                                                                           
    0.0           79505          25          3180.2            1118            8222  fclose                                                                          
    0.0           26541           8          3317.6            1035           16514  fcntl                                                                           
    0.0           26188           5          5237.6            2905            7501  open                                                                            
    0.0           15822           2          7911.0            5739           10083  socket                                                                          
    0.0           15471           2          7735.5            3893           11578  pthread_cond_signal                                                             
    0.0            9526           1          9526.0            9526            9526  pipe2                                                                           
    0.0            7645           1          7645.0            7645            7645  connect                                                                         
    0.0            5039           2          2519.5            1490            3549  fwrite                                                                          
    0.0            3818           1          3818.0            3818            3818  bind                                                                            
    0.0            2051           1          2051.0            2051            2051  listen                                                                          




Generating NVTX Push-Pop Range Statistics...
NVTX Push-Pop Range Statistics (nanoseconds)
```