#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_SIZE 20


/**
 *  const int H_out = (H - K)/S + 1;
 *  const int W_out = (W - K)/S + 1;
    dim3 dimGrid(ceil(W_out/float(BLOCK_SIZE)), ceil(H_out/float(BLOCK_SIZE)), M * B);
 *  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
 * It means for each image and each feature map, we have a grid of blocks. since we have strides, 
 * we divide works after strides into different blocks.
*/
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, 
    const int B, const int M, const int C, const int H, const int W, const int K,const int S) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */


    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    // pick a way to divide work for image and feature map
    int b = blockIdx.x % B;
    int m = blockIdx.y % M;

    int bx = blockIdx.x / B;
    int by = blockIdx.y / M;

    int w = bx * blockDim.x + threadIdx.x;
    int h = by * blockDim.y + threadIdx.y;

    // load to shared memory will be useless if stride size is big.
    // so we directly use global memory

    float sum = 0.0f;

    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if (h * S + p < H && w * S + q < W) {
                    sum += in_4d(b, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
                }
            }
        }
    }

    if(h < H_out && w < W_out)
        out_4d(b, m, h, w) = sum;


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


/**
 * @brief      Performs the forward pass of a convolutional layer
 * @param      host_output  The host output output[b][m][h][w] means 
 *  the value at output[h][w] of the mth feature map of the bth image
 * @param      host_input   The host input input[b][c][h * Stride + p][w * Stride + q] means
 *  the value at input[h * Stride + p][w * Stride + q] of the cth feature map of the bth image
 * @param      host_mask    The host mask used to compute the convolution
 * @param      device_output_ptr  The device output
 * @param      device_input_ptr   The device input
 * @param      device_mask_ptr    The device mask
 * 
 * Function paramters:
	  output - output
	  input - input
	  k - kernel
	  B - batch_size (number of images in x)
	  M - number of output feature maps
	  C - number of input feature maps
	  H - input height dimension
	  W - input width dimension
	  K - kernel height and width (K x K)
	  S - stride step length
*/
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, 
    const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, 
    const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // Allocate device memory for the output, input and mask
    cudaMalloc(device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc(device_mask_ptr, M * C * K * K * sizeof(float));
    cudaMalloc(device_output_ptr, B * M * H_out * W_out * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, 
const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    // printf("%d %d\n", H_out, W_out);
    dim3 dimGrid(ceil(1.0 * W_out/float(BLOCK_SIZE)) * B , ceil(1.0 * H_out/float(BLOCK_SIZE)) * M, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, 
    float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host

    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
