// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                      \
    do                                                                     \
    {                                                                      \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

__global__ void scan(float *input, float *output, int len, float *lastElement)
{
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from the host
    __shared__ float T[2 * BLOCK_SIZE];

    // first, load the elements into shared memory
    int tx = threadIdx.x;
    int start = 2 * blockIdx.x * blockDim.x;

    if (start + tx < len)
        T[tx] = input[start + tx];
    else
        T[tx] = 0.0;

    if (start + tx + BLOCK_SIZE < len)
        T[tx + BLOCK_SIZE] = input[start + tx + BLOCK_SIZE];
    else
        T[tx + BLOCK_SIZE] = 0.0;

    __syncthreads();

    // Build Balance Tree (reduction step)
    int stride = 1;
    while (stride < 2 * BLOCK_SIZE)
    {
        int index = (tx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE && index - stride >= 0)
        {
            T[index] += T[index - stride];
        }
        stride *= 2;
        __syncthreads();
    }

    // Build Balance Tree (post reduction step)
    stride = BLOCK_SIZE / 2;
    while (stride > 0)
    {
        int index = (tx + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE)
        {
            T[index + stride] += T[index];
        }
        stride /= 2;
        __syncthreads();
    }
    // Copy the result to output
    if (start + tx < len)
    {
        output[start + tx] = T[tx];
    }
    if (start + tx + BLOCK_SIZE < len)
    {
        output[start + tx + BLOCK_SIZE] = T[tx + BLOCK_SIZE];
    }
    if(tx == 0)
        lastElement[blockIdx.x] = T[2 * BLOCK_SIZE - 1];
}

__global__ void add(float *input, float *output, float *lastElement, int len)
{
    int tx = threadIdx.x;
    int start = 2 * blockIdx.x * blockDim.x;
    if (start + tx < len)
    {
        output[start + tx] += lastElement[blockIdx.x];
    }
    if (start + tx + BLOCK_SIZE < len)
    {
        output[start + tx + BLOCK_SIZE] += lastElement[blockIdx.x];
    }
}

int main(int argc, char **argv)
{
    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float *)malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
          numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");


    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                       cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");


    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(1.0 * numElements / (BLOCK_SIZE * 2)), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    float *host_lastElement = (float *)malloc(DimGrid.x * sizeof(float));
    float* device_lastElement;
    cudaMalloc((void **)&device_lastElement, DimGrid.x * sizeof(float));

    scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements, device_lastElement);

    cudaDeviceSynchronize();

    cudaMemcpy(host_lastElement, device_lastElement, DimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);

    // In CPU, we maintain an array recording the last element of each block (sum of all elements in the block)

    for (int i = 1; i < DimGrid.x; i++)
    {
        host_lastElement[i] += host_lastElement[i - 1];
    }
    for (int i = DimGrid.x - 1; i > 0; i--)
    {
        host_lastElement[i] = host_lastElement[i - 1];
        // printf("%f\n", host_lastElement[i]);
    }
    host_lastElement[0] = 0.0;

    cudaMemcpy(device_lastElement, host_lastElement, DimGrid.x * sizeof(float), cudaMemcpyHostToDevice);


    // add the last element of each block to the corresponding block

    add<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, device_lastElement, numElements);

    cudaDeviceSynchronize();
    
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                       cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
