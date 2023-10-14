#include <wb.h>
#include <fstream>
#include <iostream>

#define wbCheck(stmt)                                              \
	do                                                             \
	{                                                              \
		cudaError_t err = stmt;                                    \
		if (err != cudaSuccess)                                    \
		{                                                          \
			wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
			wbLog(ERROR, "Failed to run stmt ", #stmt);            \
			return -1;                                             \
		}                                                          \
	} while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_DIM 8

//@@ Define constant memory for device kernel here

__constant__ float deviceKernelConstant[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

/**
 * 3d convolution kernel
 * @brief : Every thread corresponds to an input element, including halo elements. Hence the 
 * number of threads are greater than the number of output elements, making some threads idle when computing.
 * However, all threads participate in loading the input elements to shared memory.
 * 
*/
__global__ void conv3d(float *input, float *output, const int z_size,
					   const int y_size, const int x_size)
{
	//@@ Insert kernel code here
	__shared__ float sharedInput[TILE_DIM + MASK_WIDTH - 1][TILE_DIM + MASK_WIDTH - 1][TILE_DIM + MASK_WIDTH - 1];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	// index the 3d inputs
	int x_i = blockIdx.x * TILE_DIM + tx - (MASK_WIDTH / 2);
	int y_i = blockIdx.y * TILE_DIM + ty - (MASK_WIDTH / 2);
	int z_i = blockIdx.z * TILE_DIM + tz - (MASK_WIDTH / 2);
	// index the 3d outputs (discard if tx / ty / tz is out of bounds)
	int x_o = blockIdx.x * TILE_DIM + tx;
	int y_o = blockIdx.y * TILE_DIM + ty;
	int z_o = blockIdx.z * TILE_DIM + tz;

	// if(!tx && !ty && !tz) {
	// 	printf("%d %d %d %d\n", x_i, y_i, z_i, (MASK_WIDTH / 2 - 1));
	// }

	// load the input elements to shared memory
	if (x_i >= 0 && x_i < x_size && y_i >= 0 && y_i < y_size && z_i >= 0 && z_i < z_size)
	{
		sharedInput[tz][ty][tx] = input[z_i * y_size * x_size + y_i * x_size + x_i];
	}
	else
	{
		sharedInput[tz][ty][tx] = 0.0f;
	}

	__syncthreads();

	// compute the output elements
	if (tx < TILE_DIM && ty < TILE_DIM && tz < TILE_DIM && x_o < x_size && y_o < y_size && z_o < z_size)
	{
		float pValue = 0.0f;
		for (int i = 0; i < MASK_WIDTH; i++)
		{
			for (int j = 0; j < MASK_WIDTH; j++)
			{
				for (int k = 0; k < MASK_WIDTH; k++)
				{
					pValue += deviceKernelConstant[i][j][k] * sharedInput[tz + i][ty + j][tx + k];
				}
			}
		}
		// if(x_o == 7 && !y_o && !z_o){
		// 	// print shared Input
		// 	for(int i = 0; i < TILE_DIM + MASK_WIDTH - 1; i++) {
		// 		for(int j = 0; j < TILE_DIM + MASK_WIDTH - 1; j++) {
		// 			for(int k = 0; k < TILE_DIM + MASK_WIDTH - 1; k++) {
		// 				printf("%f ", sharedInput[i][j][k]);
		// 			}
		// 			printf("\n");
		// 		}
		// 		printf("\n");
		// 	}
		// 	printf("%d %d %d %f\n", x_o, y_o, z_o, pValue);
		// }
		output[z_o * y_size * x_size + y_o * x_size + x_o] = pValue;
	}


}

int main(int argc, char *argv[])
{
	wbArg_t args;
	int z_size;
	int y_size;
	int x_size;
	int inputLength, kernelLength;
	float *hostInput;
	float *hostKernel;
	float *hostOutput;
	float *deviceInput;
	float *deviceOutput;

	args = wbArg_read(argc, argv);

	// Import data
	hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostKernel =
		(float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
	hostOutput = (float *)malloc(inputLength * sizeof(float));

	// First three elements are the input dimensions
	z_size = hostInput[0];
	y_size = hostInput[1];
	x_size = hostInput[2];
	wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
	assert(z_size * y_size * x_size == inputLength - 3);
	assert(kernelLength == 27);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//@@ Allocate GPU memory here
	// Recall that inputLength is 3 elements longer than the input data
	// because the first  three elements were the dimensions
	wbTime_stop(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInput, z_size * y_size * x_size * sizeof(float));
	cudaMalloc((void **)&deviceOutput, z_size * y_size * x_size * sizeof(float));
	

	wbTime_start(Copy, "Copying data to the GPU");
	//@@ Copy input and kernel to GPU here
	// Recall that the first three elements of hostInput are dimensions and
	// do
	// not need to be copied to the gpu
	wbTime_stop(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInput, hostInput + 3, z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
	// std::cout << hostInput[3] << std::endl;
	// copy kernel to constant memory
	cudaMemcpyToSymbol(deviceKernelConstant, hostKernel, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float));

	// std::ofstream outfile1;
	// outfile1.open("kernel.txt");
	// for(int i = 0; i < MASK_WIDTH; i++) {
	//      for(int j = 0; j < MASK_WIDTH; j++) {
	// 		for(int k = 0; k < MASK_WIDTH; k++) {
	// 			outfile1 << hostKernel[k * MASK_WIDTH * MASK_WIDTH + j * MASK_WIDTH + i] << " ";
	// 		}
	// 		outfile1 << std::endl;
	//      }
	//      outfile1 << std::endl;
	// }
	// outfile1.close();

	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ Initialize grid and block dimensions here
	dim3 DimGrid(ceil(x_size / (1.0 * TILE_DIM)), ceil(y_size / (1.0 * TILE_DIM)), ceil(z_size / (1.0 * TILE_DIM)));
	dim3 DimBlock(TILE_DIM + MASK_WIDTH - 1, TILE_DIM + MASK_WIDTH - 1, TILE_DIM + MASK_WIDTH - 1);

	//@@ Launch the GPU kernel here
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);


	wbTime_start(Copy, "Copying data from the GPU");
	//@@ Copy the device memory back to the host here
	// Recall that the first three elements of the output are the dimensions
	// and should not be set here (they are set below)
	cudaMemcpy(hostOutput + 3, deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	// Set the output dimensions for correctness checking
	hostOutput[0] = z_size;
	hostOutput[1] = y_size;
	hostOutput[2] = x_size;
	wbSolution(args, hostOutput, inputLength);

	// std::ofstream outfile;
	// outfile.open("output.txt");
	// outfile << z_size << " " << y_size << " " << x_size << std::endl;
	// for(int i = 0; i < x_size; i++) {
	//      for(int j = 0; j < y_size; j++) {
	// 		for(int k = 0; k < z_size; k++) {
	// 			outfile << hostOutput[k * y_size * x_size + j * x_size + i + 3] << " ";
	// 		}
	// 		outfile << std::endl;
	//      }
	//      outfile << std::endl;
	// }
	// outfile.close();

	// Free device memory
	cudaFree(deviceInput);
	cudaFree(deviceOutput);

	// Free host memory
	free(hostInput);
	free(hostOutput);
	return 0;
}
