// Histogram Equalization

#include <wb.h>
#include <cstdio>
#include <iostream>

#define HISTOGRAM_LENGTH 256
#define TILE_WIDTH 16
#define NUM_CHANNELS 3

//@@ insert code here

/** GPU Kernel
 * convert image of floats to image of unsigned char
 * Input:
 * - float *inputImage
 * - int imageWidth
 * - int imageHeight
 * - int imageChannels
 * - unsigned char *outputImage
*/

__global__ void floatToUchar(float *inputImage, int imageWidth, int imageHeight, int imageChannels, 
							unsigned char *outputImage) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < imageWidth && y < imageHeight && z < imageChannels)
	{
		int index = (y * imageWidth + x) * imageChannels + z;
		outputImage[index] = (unsigned char)(255 * inputImage[index]);
	}
}

/** GPU Kernel
 * convert image from RGB to GrayScale
 * Input:
 * - unsigned char *inputImage
 * - int imageWidth
 * - int imageHeight
 * - unsigned char *outputImage
*/

__global__ void rgbToGrayScale(unsigned char *inputImage, int imageWidth, int imageHeight,
							unsigned char *outputImage) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < imageWidth && y < imageHeight)
	{
		int index = (y * imageWidth + x);
		outputImage[index] = (unsigned char) (0.21f * inputImage[index * NUM_CHANNELS] 
								+ 0.71f * inputImage[index * NUM_CHANNELS + 1] 
								+ 0.07f * inputImage[index * NUM_CHANNELS + 2]
							);
	}
}

/** GPU Kernel
 * compute histogram of image
 * Input:
 * - unsigned char *grayScaleImage
 * - int imageWidth
 * - int imageHeight
 * - unsigned int *histogram
*/

__global__ void computeHistogram(unsigned char *grayScaleImage, int imageWidth, int imageHeight, 
								unsigned int *histogram) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size = imageWidth * imageHeight;
	__shared__ unsigned int privateHistogram[HISTOGRAM_LENGTH];

	if (index < size)
	{
		int privateIndex = threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		// initialize private histogram
		if(privateIndex < HISTOGRAM_LENGTH)
			privateHistogram[privateIndex] = 0;
		
		__syncthreads();

		while (index < size)
		{
			atomicAdd(&(privateHistogram[grayScaleImage[index]]), 1);
			index += stride;
		}
		
		__syncthreads();

		// merge private histogram to global histogram
		if(privateIndex < HISTOGRAM_LENGTH)
			atomicAdd(&(histogram[privateIndex]), privateHistogram[privateIndex]);
	}
}

/** CPU function
 * compute cdf of histogram
 * p(x) = x / (width * height)
 * Input:
 * - unsigned int *histogram
*/

void computeCdf(unsigned int *histogram, int width, int height, float *cdf) {
	cdf[0] = (float)histogram[0] / (width * height);
	unsigned int sum = histogram[0];
	// printf("cdf[0] = %f\n", cdf[0]);
	for (int i = 1; i < HISTOGRAM_LENGTH; i++)
	{
		sum += histogram[i];
		cdf[i] = cdf[i - 1] + (float)histogram[i] / (width * height);
		// printf("cdf[%d] = %f\n", i, cdf[i]);
	}
	// printf("%u\n", sum);

}

/** GPU kernel
 * apply histogram equalization, casting them back to float
 * Input:
 * - unsigned char *grayScaleImage
 * - int imageWidth
 * - int imageHeight
 * - int imageChannels
 * - float *cdf
 * - float *outputImage
 * 
* def correct_color(val) 
		return clamp(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0, 255.0)
	end

	def clamp(x, start, end)
		return min(max(x, start), end)
	end
*/

__global__ void histogramEqualization(unsigned char *grayScaleImage, 
									int imageWidth, int imageHeight, int imageChannels,
									float *cdf, float *outputImage) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < imageWidth && y < imageHeight && z < imageChannels)
	{
		int index = (y * imageWidth + x) * imageChannels + z;
		float pVal;
		pVal = 255.0 * (cdf[grayScaleImage[index]] - cdf[0]) / (1.0 - cdf[0]);
		if (pVal > 255)
			pVal = 255;
		// convert back to unsigned char
		unsigned char rpVal = (unsigned char)pVal;
		outputImage[index] = (float)rpVal / 255.0;
	}
}





int main(int argc, char **argv)
{
	wbArg_t args;
	int imageWidth;
	int imageHeight;
	int imageChannels;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	// float *refhostOutputImageData;
	const char *inputImageFile;
	// const char *refOutputImageFile;
	// wbImage_t refOutputImage;

	//@@ Insert more code here

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);
	// refOutputImageFile = wbArg_getInputFile(args, 1);

	wbTime_start(Generic, "Importing data and creating memory on host");
	inputImage = wbImport(inputImageFile);
	// refOutputImage = wbImport(refOutputImageFile);
	// refhostOutputImageData = wbImage_getData(refOutputImage);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	//@@ insert code here

	// allocate kernel memory
	float *deviceInputImageData;
	float *deviceOutputImageData;
	unsigned char *deviceInputUCharImageData;
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceInputUCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));

	// copy host memory to device
	cudaMemcpy(deviceInputImageData, hostInputImageData, 
				imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	
	// invoke kernel
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, imageChannels);
	dim3 dimGrid((imageWidth - 1) / TILE_WIDTH + 1, 
				(imageHeight - 1) / TILE_WIDTH + 1, 1);
	// convert image from float to unsigned char
	floatToUchar<<<dimGrid, dimBlock>>>(deviceInputImageData, imageWidth, imageHeight, imageChannels, 
										deviceInputUCharImageData);

	cudaDeviceSynchronize();
	// convert image from RGB to GrayScale
	// reset kernel dimension
	dimBlock.z = 1;
	unsigned char *deviceGrayScaleImageData;
	cudaMalloc((void **)&deviceGrayScaleImageData, imageWidth * imageHeight * sizeof(unsigned char));
	rgbToGrayScale<<<dimGrid, dimBlock>>>(deviceInputUCharImageData, imageWidth, imageHeight,
										deviceGrayScaleImageData);

	cudaDeviceSynchronize();

	// compute histogram

	dim3 hist_dimBlock(HISTOGRAM_LENGTH, 1, 1);
	dim3 hist_dimGrid((imageWidth * imageHeight - 1) / (HISTOGRAM_LENGTH * HISTOGRAM_LENGTH ) + 1, 1, 1);

	unsigned int *deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
	cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
	computeHistogram<<<hist_dimGrid, hist_dimBlock>>>(deviceGrayScaleImageData, imageWidth, imageHeight, deviceHistogram);

	cudaDeviceSynchronize();

	// in CPU, we compute cdf
	unsigned int *hostHistogram = (unsigned int *)malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
	cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	float *hostCdf = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));
	
	computeCdf(hostHistogram, imageWidth, imageHeight, hostCdf);


	// in GPU, we apply histogram equalization
	// reset kernel dimension
	dimBlock.z = imageChannels;
	float *deviceCdf;
	cudaMalloc((void **)&deviceCdf, HISTOGRAM_LENGTH * sizeof(float));
	cudaMemcpy(deviceCdf, hostCdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaDeviceSynchronize();

	histogramEqualization<<<dimGrid, dimBlock>>>(deviceInputUCharImageData, imageWidth, imageHeight, imageChannels,
												deviceCdf,  deviceOutputImageData);

	cudaDeviceSynchronize();
	// copy device memory to host
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, 
				imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

	// printf("%f %f\n", hostOutputImageData[0], refhostOutputImageData[0]);

	wbSolution(args, outputImage);

	//@@ insert code here

	// free memory
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceGrayScaleImageData);
	cudaFree(deviceInputUCharImageData);
	cudaFree(deviceHistogram);
	cudaFree(deviceCdf);
	free(hostHistogram);
	free(hostCdf);
	return 0;
}
