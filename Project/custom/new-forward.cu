#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_SIZE 16

#define IS_TEST_1(B,M,C,H,W,K,S)  (B==1 && M == 3 && C == 3 && H == 224 && W == 224 && K == 3 && S == 1)

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, 
    const int B, const int M, const int C, const int H, const int W, const int K,const int S) {
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP
    // printf("!\n");
	// __shared__ float MM[BLOCK_SIZE][BLOCK_SIZE];
	// __shared__ float NN[BLOCK_SIZE][BLOCK_SIZE];


    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    int numARows = H_out * W_out;
    int numAColumns = C * K * K;
    int numBRows = C * K * K;
    int numBColumns = M;
    // int numCRows = numARows;
    // int numCColumns = numBColumns;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]


	int bx = blockIdx.x; int by = blockIdx.y ; int bz = blockIdx.z;
	int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

	int Row = bx * BLOCK_SIZE + tx;
	int Col = (by * BLOCK_SIZE + ty) % M;

	int b = (by * BLOCK_SIZE + ty) / M;
    int m = Col;

    
    // without using shared memory
    if( Row < numARows && Col < numBColumns && (by * BLOCK_SIZE + ty) < M * B) {

        out_4d(b, m , Row / W_out , Row % W_out) = 0;
        
        float Pvalue = 0;
        for(int k = 0; k < numAColumns; k++) {
            const float* A = input  + b * numAColumns * numARows;
            const float* B = mask;
            Pvalue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
        }
        out_4d(b, m , Row / W_out , Row % W_out) = Pvalue;
    }

    
	// float Pvalue = 0;
	// for(int kk = 0; kk < (numAColumns - 1) / BLOCK_SIZE + 1; kk++) {
    //     const float* AA = input + b * numAColumns * numARows;
    //     const float* BB = mask;

	// 	if(Row < numARows && kk * BLOCK_SIZE + ty < numAColumns)
	// 		MM[tx][ty] = AA[Row * numAColumns + kk * BLOCK_SIZE + ty];
	// 	else
	// 		MM[tx][ty] = 0.0;

	// 	if(kk * BLOCK_SIZE + tx < numBRows && (by * BLOCK_SIZE + ty) < M * B)
	// 		NN[tx][ty] = BB[(kk * BLOCK_SIZE + tx) * numBColumns + Col];
	// 	else
	// 		NN[tx][ty] = 0.0;

	// 	__syncthreads(); // we first wait until the two blocks are loaded
    //     // this can be optimized using tensor core

	// 	for(int k = 0; k < BLOCK_SIZE; ++k)
	// 		Pvalue += MM[tx][k] * NN[k][ty];

	// 	__syncthreads();
	// }

	// if(Row < numARows && (by * BLOCK_SIZE + ty) < M * B) {
    // 	out_4d(b, m , Row / W_out , Row % W_out) = Pvalue;
    // }

    #undef out_4d
}

/**
 * Unroll the input matrix to have K*K*C columns and w_out*h_out rows
 * 
*/
static void im2col(float* output, const float *input, float* mask_o, const float* mask_i,
                    const int B, const int M, const int C, const int H, 
                    const int W, const int K,const int S) {
    
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define out_6d(i5, i4, i3, i2, i1, i0) output[ \
        (i5) * C * K * K * ((H - K)/S + 1)*((W - K)/S + 1) + \
        (i4) * (C * K * K *((W - K)/S + 1)) + \
        (i3) * (C * K * K ) + \
        (i2) * K * K + \
        (i1) * K + \
        (i0) \
    ]
    #define old_mask_4d(i3, i2, i1, i0) mask_i[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d(i3, i2, i1, i0) mask_o[(i3) * (K * K * M) + (i2) * (K * M) + (i1) * (M) + i0]



    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // this version doesn't inoke kernel for unrolling

    // unroll the input and assign it to output
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    for (int p = 0; p < K; p++) {
                        for (int q = 0; q < K; q++) {
                            out_6d(b, h, w, c, p, q) = in_4d(b, c, h * S + p, w * S + q);
                        }
                    }
                }
            }
        }
    }
    // if(IS_TEST_1(B,M,C,H,W,K,S)) {
    //     if(out_6d(0, 0, 16, 0, 1,2)) {
    //         printf("out_6d(%d, %d, %d, %d, %d, %d) = %f\n", 
    //             0, 0, 16, 0, 1,2, out_6d(0, 0, 16, 0, 1,2));
    //     }
    // }

    // re-oraganize the mask
    for (int m = 0; m < M; m++) {
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    mask_4d(c, p, q, m) = old_mask_4d(m, c, p, q);
                }
            }
        }
    }

    #undef out_6d
    #undef in_4d
    #undef mask_4d
    #undef old_mask_4d
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

    // const int H_out = (H - K)/S + 1;
    // const int W_out = (W - K)/S + 1;

    // Allocate device memory for the output, input and mask
//     cudaMalloc(device_output_ptr, B * M * H_out * W_out * sizeof(float));

// // #if (CUR_VERSION == IMPL_BASELINE)
//     cudaMalloc(device_input_ptr, B * C * H * W * sizeof(float));
//     cudaMalloc(device_mask_ptr, M * C * K * K * sizeof(float));

//     cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);


    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // Allocate device memory for the output, input and mask
    cudaMalloc(device_output_ptr, B * M * H_out * W_out * sizeof(float));
    
    // unroll the input and assign it to output
    float *device_unroll_input;
    float *device_unroll_mask;
    float *host_unroll_input;
    float *host_unroll_mask;
    host_unroll_input = (float*)malloc(B * C * K * K * H_out * W_out * sizeof(float));
    host_unroll_mask = (float*)malloc(M * C * K * K * sizeof(float));
    cudaMalloc(&device_unroll_mask, M * C * K * K * sizeof(float));
    cudaMalloc(&device_unroll_input, B * C * K * K * H_out * W_out * sizeof(float));
    
    im2col(host_unroll_input, host_input, host_unroll_mask, host_mask, B, M, C, H, W, K, S);

    // if(IS_TEST_1(B,M,C,H,W,K,S)) {
    //     for(int i = 0; i < 20; i++) {
    //         printf("host_unroll_input[%d] = %f\n", i, host_unroll_input[i]);
    //     }
    //     for(int i = 0; i < 20; i++) {
    //         printf("host_unroll_mask[%d] = %f\n", i, host_unroll_mask[i]);
    //     }
    // }


    
    cudaMemcpy(device_unroll_input, host_unroll_input, B * C * K * K * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_unroll_mask, host_unroll_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    
    free(host_unroll_input);
    free(host_unroll_mask);
    *device_input_ptr = device_unroll_input; // now the input pointer has been swapped to unrolled input
    *device_mask_ptr = device_unroll_mask; // now the mask pointer has been swapped to unrolled mask
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, 
const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    // printf("%d %d\n", H_out, W_out);
    dim3 dimGrid(ceil(1.0 * H_out * W_out / BLOCK_SIZE), B * ceil(1.0 * M / BLOCK_SIZE), 1);
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // printf("%d %d %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    // printf("%d %d %d\n", dimBlock.x, dimBlock.y, dimBlock.z);

    

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

    cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, 
    float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host

    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    // if(IS_TEST_1(B,M,C,H,W,K,S)) {
    //     for(int i = 0; i < 100; i++) {
    //         if(host_output[i] != 1) {
    //             printf("host_output[%d] = %f\n", i, host_output[i]);
    //         }
    //     }
    // }

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

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